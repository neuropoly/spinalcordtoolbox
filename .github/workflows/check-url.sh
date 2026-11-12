#!/usr/bin/env bash

if [ $# -ne 2 ]; then
    exit 1;
fi

# argument parsing
filename=$1
URL=$2
echo -e "\n${filename}: Checking URL ${URL}..."

###############################################################################
#                  Run `curl` to resolve URLs and status code                 #
###############################################################################

# --head: Sends a HEAD request. We use this to be good netizens, since we only need the header to check the response code.
# --silent: Hides the curl progress bar, which is unnecessary noise when testing >600 urls.
# --insecure: Skip SSL verification. Since we're only checking the headers, this should be safe to do. (https://curl.se/docs/sslcerts.html)
CURL_ARGS_HEAD=(--head --silent --insecure)
CURL_ARGS_GET=(--silent --insecure)
# Explicitly write out the HTTP code to stdout, while redirecting the response output to /dev/null
HTTP_CODE_ONLY=(--write-out '%{http_code}' --output /dev/null)
HTTP_CODE_PLUS_BOTH_URLS=(--write-out '%{http_code}|%{url}|%{url_effective}' --output /dev/null --location)
# Override default behavior (exponential backoff + 10m limit) since we don't need that many retries
# We still keep a 5m limit, though, because --retry respects the Retry-After field, which may be greater than 30s.
RETRY_ARGS=(--retry 2 --retry-delay 30 --retry-max-time 300 --retry-all-errors)

# Special-case `doi.org` links: DOIs are permanent identifiers that redirect to a secondary
# (potentially flaky or access-restricted) URL. Rather than fetching that URL, we only check
# that doi.org resolves the DOI to some other host. This validates that the DOI is registered.
DOI_RE='^https?://(dx\.|www\.)?doi\.org/'
if [[ "$URL" =~ ${DOI_RE}10\. ]]; then
    # Force https so doi.org's own http->https redirect can't make a bad DOI look valid.
    https_url="https://${URL#*://}"

    # No `--insecure`: we're trusting the Location header, so we want TLS verification.
    # `%{redirect_url}` is the Location header; without `--location`, curl doesn't follow it.
    read -r doi_code redirect_url < <(
        curl --head --silent "${RETRY_ARGS[@]}" \
             --write-out '%{http_code} %{redirect_url}\n' --output /dev/null -- "$https_url"
    )

    if [[ -n "$redirect_url" && ! "$redirect_url" =~ $DOI_RE ]]; then
        echo "(redirect) $https_url --> $redirect_url ($filename)" >> valid_urls.txt
        echo -e "$filename: \x1B[32m✅  OK - doi.org redirects to: $redirect_url  \x1B[0m"
        exit 0
    else
        echo -e "(\x1B[31m$doi_code\x1B[0m) $https_url ($filename)" >> invalid_urls.txt
        echo -e "$filename: \x1B[31m⛔ Error - doi.org did not redirect elsewhere - code: $doi_code for URL $https_url \x1B[0m"
        exit 1
    fi
fi

# Do an initial `--head` check to figure out both the original and redirected URLs,
# as well as the final status code after any redirections.
IFS='|' read -r status_code original_url effective_url < <(
  curl "${CURL_ARGS_HEAD[@]}" "${HTTP_CODE_PLUS_BOTH_URLS[@]}" "${RETRY_ARGS[@]}" -- "$URL"
)

# Check to see if either URLs are in the exclusion list, which includes:
# - `pipeline-hemis` -> private repository, will 404 (expected)
# - `.ru` -> Russian domains, which don't play nicely with curl'ing from GitHub's servers
# - `ieeexplore.ieee.org` -> oddly returns a "418 - I'm a teapot" error code instead of 403
# - `thejns.org` -> returns 405 for both HEAD and GET requests, rather than the 403 it used to return (curling locally still gives 403, oddly)
# - `sct_tutorial_data` -> partial URL, which is used in our documentation's `conf.py` file
# - `biorxiv.org` -> throws 429 for any requests
if [[ "$original_url + $effective_url" =~ pipeline-hemis|\.ru|ieeexplore\.ieee\.org|thejns\.org|sct_tutorial_data|biorxiv\.org ]]; then
    echo -e "$filename: \x1B[33m⚠️ Warning - Skipping: $URL --> $LOCATION\x1B[0m"
    exit 0
fi

# If the original URL redirects, then emit a log message
if [[ "$original_url" != "$effective_url" ]];then
    # Run without `--location` to get the status code for the original URL (to determine the type of redirect)
    original_code=$(curl "${CURL_ARGS_HEAD[@]}" "${HTTP_CODE_ONLY[@]}" -- "$original_url")
    echo "($original_code) $original_url ($filename)" >> redirected_urls.txt
    echo -e "$filename: \x1B[33m⚠️ Warning - Redirection - code: $original_code for URL $original_url --> $effective_url \x1B[0m"
fi

# Check for "405 Method Not Allowed" error code (https://developer.mozilla.org/en-US/docs/Web/HTTP/Status/405)
# From https://http.dev/405: "returned by the server to indicate that the resource specified by the request exists
#                             but the requested HTTP method is not allowed."
# We encountered this error for the 'https://pmc.ncbi.nlm.nih.gov' domain, which explicitly defines `allow: GET`
# So, retry without `--head` (i.e. a GET request). We could do this by default, but it's more expensive to do.
if [[ $status_code -eq 405 ]];then
    echo -e "$filename: \x1B[33m⚠️ Warning - HEAD request not allowed - code: $status_code for URL $effective_url\x1B[0m"
    status_code=$(curl "${CURL_ARGS_GET[@]}" "${HTTP_CODE_ONLY[@]}" "${RETRY_ARGS[@]}" -- "$effective_url")
fi

###############################################################################
#                             Analyze status code                             #
###############################################################################

# Check for success
if [[ $status_code -ge 200 && $status_code -le 299 ]];then
    echo "($status_code) $effective_url ($filename)" >> valid_urls.txt
    echo -e "$filename: \x1B[32m✅  OK status code: $status_code for domain $effective_url  \x1B[0m"
    exit 0
fi

# Check for "403 Forbidden" error code (https://developer.mozilla.org/en-US/docs/Web/HTTP/Status/403)
# From https://http.dev/403: "The client does not have access to the requested resource."
# - Due to the rise of AI and LLMs, many sites are now blocking automated access to their content.
# - Often there is no way to bypass this, so in the past we hardcoded the domains to an exclusion list.
#   However, manually editing the blacklist is not a sustainable solution, so we now automatically filter
#   out any sites that raise 403 with just a warning instead of an error.
# - Note: Allowing 403 responses covers the following domains:
#         'twitter.com', 'spiedigitallibrary.org', 'sciencedirect.com', 'wiley.com', 'sagepub.com',
#         'ncbi.nlm.nih.gov', 'oxfordjournals.org', 'docker.com', 'liebertpub.com', 'tandfonline.com',
#         'pnas.org', 'neurology.org', 'academic.oup.com', 'science.org', 'pubs.rsna.org', 'direct.mit.edu',
#         'ajnr.org', 'bmj.com', and 'biorxiv.org'
if [[ $status_code -eq 403 ]];then
    echo "($status_code) $effective_url ($filename)" >> valid_urls.txt
    echo -e "$filename: \x1B[33m⚠️ Warning - Forbidden - status code: $status_code for domain $effective_url \x1B[0m"
    exit 0
fi

# Check for "406 Not Acceptable" error code (https://developer.mozilla.org/en-US/docs/Web/HTTP/Status/406)
# From https://http.dev/406: "In practice, this error is rarely used because the server supplies a default
#                             representation instead."
# We encountered this error for the 'Radiopaedia.org' domain.
# After some searching, I believe there are two main causes of this error:
#   - Overly strict 'Accept-*' headers. (In our case, the default header for curl, wget, etc. specifies "accept: */*",
#                                        which is about as open as can be. So I don't think this is the culprit.)
#   - User-agent strings (see e.g. https://stackoverflow.com/a/10043347). (Changing this locally still resulted in 406.)
#
# My best guess is that this due to either A) misconfigured CloudFlare B) a way to prevent LLMs from scraping content.
# So, for now, we just filter out this response, as there's a good chance that this is still accessible via browsers.
if [[ $status_code -eq 406 ]];then
    echo "($status_code) $effective_url ($filename)" >> valid_urls.txt
    echo -e "$filename: \x1B[33m⚠️ Warning - Not Acceptable - status code: $status_code for domain $effective_url  \x1B[0m"
    exit 0
fi

# Report failure
echo -e "(\x1B[31m$status_code\x1B[0m) $effective_url ($filename)" >> invalid_urls.txt
echo -e "$filename: \x1B[31m⛔ Error status code: $status_code for URL: $effective_url \x1B[0m"
