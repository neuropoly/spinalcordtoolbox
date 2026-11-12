"""
Deal with SCT dataset metadata

Copyright (c) 2018 Polytechnique Montreal <www.neuro.polymtl.ca>
License: see the file LICENSE
"""

import io
import os
import re
from operator import itemgetter

from spinalcordtoolbox.utils.shell import parse_num_list


class InfoLabel(object):
    """
    Class representing data available in info_label.txt meta-data files, which
    are used to tie together several NIFTI files covering the same volume,
    each containing a layer.

    The info_label.txt file contains at least one `IndivLabels` section, which
    lists layer label id, layer label name, and layer filename (path relative to the
    info_label.txt file).

    Another possible section is `CombinedLabels` which is creating new labels
    (ids and names) by combining several labels from `IndivLabels`.

    Another possible section is `MAPLabels` which contains clusters used for
    the first step of the MAP estimation ("for advanced users only").

    The file is text, UTF-8 encoded; see the loading/saving code for the detailed
    file layout.
    """
    def __init__(self, indiv_labels=None, combined_labels=None, clusters_apriori=None):
        """
        :param indiv_labels: list of 3-tuple with (id, name, filename)
        :param combined_labels: list of 3-tuple with (id, name, ids)
        :param clusters_apriori: list of 2-tuple with (name, ids)
        """
        self._indiv_labels = list() if indiv_labels is None else indiv_labels
        self._combined_labels = list() if combined_labels is None else combined_labels
        self._clusters_apriori = list() if clusters_apriori is None else clusters_apriori

    def load(self, file, parent=None, verify=True):
        """
        Load contents from file
        :param file: input filename or file-like object to load from
        """

        # reset contents
        self.__init__()

        if isinstance(file, str):
            parent = os.path.dirname(file)
            file = io.open(file, "rb")

        section = ""
        for idx_line, line in enumerate(file):
            line = line.rstrip().decode("utf-8")
            if line == "":
                continue

            # update section index
            m_sec = re.match(r"^# Keyword=(?P<kw>IndivLabels|CombinedLabels|MAPLabels)\s*(?P<comment>.*)$", line)
            if m_sec is not None:
                section = m_sec.group("kw")
                continue

            m_comment = re.match(r"#.*", line)
            if m_comment is not None:
                continue

            if section == 'IndivLabels':
                m = re.match(r"^(?P<id>\d+), (?P<name>.*), (?P<filename>.*)$", line)
                if m is None:
                    raise ValueError(f"Unexpected at line {idx_line+1}, in IndivLabels section: {line}")

                _id = int(m.group("id"))
                _name = m.group("name")
                _filename = m.group("filename")

                if verify and parent is not None:
                    if not os.path.exists(os.path.join(parent, _filename)):
                        raise ValueError(f"Unexpected at line {idx_line+1}, specifying file {_filename} which doesn't exist: {line}")

                self._indiv_labels.append((_id, _name, _filename))

            elif section == 'CombinedLabels':
                m = re.match(r"^(?P<id>\d+), (?P<name>.*), (?P<group>.*)$", line)
                if m is None:
                    raise ValueError(f"Unexpected at line {idx_line+1}, in CombinedLabels section: {line}")

                _id = int(m.group("id"))
                _name = m.group("name")

                try:
                    _group = parse_num_list(m.group("group"))
                except ValueError as e:
                    raise ValueError(f"Unexpected at line {idx_line+1}: {e} in line: {line}")

                self._combined_labels.append((_id, _name, _group))

            elif section == 'MAPLabels':
                m = re.match(r"^(?P<name>.*), (?P<group>.*)$", line)
                if m is None:
                    raise ValueError(f"Unexpected at line {idx_line+1}, in MAPLabels section: {line}")

                _name = m.group("name")

                try:
                    _group = parse_num_list(m.group("group"))
                except ValueError as e:
                    raise ValueError(f"Unexpected at line {idx_line+1}: {e} in line: {line}")

                self._clusters_apriori.append((_name, _group))

            else:
                raise ValueError(f"Unexpected at line {idx_line+1}, unparsed data: {line}")

    def save(self, file, header=None):
        """
        Save contents to file
        :param file: output filename or file-like object
        """

        if isinstance(file, str):
            file = io.open(file, "wb")

        if not self._indiv_labels:
            raise RuntimeError("Nothing to dump (no labels)")

        def w(x):
            # Writer in bytes encoding (for py3k compatibility)
            file.write(x.encode() + b'\n')

        if header is not None:
            w(f"# {header}")

        w("# Keyword=IndivLabels (Please DO NOT change this line)")
        w("# ID, name, file")
        for _id, _name, _filename in self._indiv_labels:
            w(f"{_id}, {_name}, {_filename}")

        if self._combined_labels:
            w("")
            w("# Combined labels")
            w("# Keyword=CombinedLabels (Please DO NOT change this line)")
            w("# ID, name, IDgroup")
            for _id, _name, _group in self._combined_labels:
                group_str = ",".join([str(x) for x in _group])  # could be shortened
                w(f"{_id}, {_name}, {group_str}")

        if self._clusters_apriori:
            w("")
            w("# Clusters used for the first step of the MAP estimation (for advanced users only)")
            w("# Keyword=MAPLabels (Please DO NOT change this line)")
            w("# Name, IDgroup")
            for _name, _group in self._clusters_apriori:
                group_str = ",".join([str(x) for x in _group])  # could be shortened
                w(f"{_name}, {group_str}")


def read_label_file(path_info_label, file_info_label):
    """Reads file_info_label (located inside label folder) and returns the information needed."""

    il = InfoLabel()
    fname_label = os.path.join(path_info_label, file_info_label)
    il.load(fname_label)

    indiv_labels_ids, indiv_labels_names, indiv_labels_files = zip(*il._indiv_labels)

    if il._combined_labels:
        combined_labels_ids, combined_labels_names, combined_labels_id_groups = zip(*il._combined_labels)
    else:
        combined_labels_ids, combined_labels_names, combined_labels_id_groups = [], [], []

    if il._clusters_apriori:
        clusters_apriori = [x[1] for x in il._clusters_apriori]
    else:
        clusters_apriori = []

    return indiv_labels_ids, indiv_labels_names, indiv_labels_files, combined_labels_ids, combined_labels_names, combined_labels_id_groups, clusters_apriori


def read_label_file_atlas(path_info_label, file_info_label):
    il = InfoLabel()
    fname_label = os.path.join(path_info_label, file_info_label)
    il.load(fname_label)
    return list(zip(*il._indiv_labels))


def get_file_label(path_label='', id_label=0, output='file'):
    """
    Get label file name given based on info_label.txt file.
    :param path_label: folder containing info_label.txt and the files
    :param id_label: (int) ID of the label to be found
    :param output: {file, filewithpath}
    :return: selected output ; if not found, raise a RuntimeError
    """

    file_info_label = 'info_label.txt'
    il = InfoLabel()
    fname_label = os.path.join(path_label, file_info_label)
    il.load(fname_label)

    for _id, _name, _file in il._indiv_labels:
        if _id == id_label:
            if output == 'file':
                return _file
            elif output == 'filewithpath':
                return os.path.join(path_label, _file)

    raise RuntimeError(f"Label ID {id_label} not found in {fname_label}")


def get_indiv_label_info(directory):
    """
    Get all individual label info (id, name, filename) in a folder
    :param directory: folder containing info_label.txt and the files
    :return: dictionary containing "id" the label IDs (int),\
                                    "name" the labels (string),\
                                    "file" the label filename (string)
    """

    file_info_label = 'info_label.txt'
    il = InfoLabel()
    fname_label = os.path.join(directory, file_info_label)
    il.load(fname_label)

    id_lst = list(map(itemgetter(0), il._indiv_labels))
    name_lst = list(map(itemgetter(1), il._indiv_labels))
    filename_lst = list(map(itemgetter(2), il._indiv_labels))

    return {'id': tuple(id_lst),
            'name': tuple(name_lst),
            'file': tuple(filename_lst)
            }
