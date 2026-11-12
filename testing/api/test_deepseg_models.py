# pytest unit tests for spinalcordtoolbox.deepseg.models

import json

import pytest

import spinalcordtoolbox.deepseg.models as deepseg_models


def test_load_crop_metadata(tmp_path, monkeypatch):
    """
    `load_crop_metadata()` should: skip the disk entirely for non-crop models; require an
    explicit, complete `crop_metadata.json` for crop models on official installs; tolerate a
    missing file only for `-custom-url` installs; and reject a file that's missing any of the
    padding keys, rather than silently falling back to `sc_crop.detect()`'s own defaults.
    """
    monkeypatch.setitem(deepseg_models.MODELS, 'not_a_crop_model', {'cropped_image': False})
    monkeypatch.setitem(deepseg_models.MODELS, 'a_crop_model', {'cropped_image': True})
    path_model = str(tmp_path)

    # Non-crop model: nothing on disk is even checked.
    assert deepseg_models.load_crop_metadata('not_a_crop_model', path_model) == {}

    # Crop model, no crop_metadata.json, official install: raises.
    with pytest.raises(RuntimeError) as e:
        deepseg_models.load_crop_metadata('a_crop_model', path_model)
    assert "no crop_metadata.json was found" in str(e.value)

    # Crop model, no crop_metadata.json, `-custom-url` install: tolerated.
    fname_source = tmp_path / "source.json"
    fname_source.write_text(json.dumps({'custom': True}))
    assert deepseg_models.load_crop_metadata('a_crop_model', path_model) == {}
    fname_source.unlink()

    # crop_metadata.json present but missing some padding keys: raises.
    fname_json = tmp_path / "crop_metadata.json"
    fname_json.write_text(json.dumps({'pad_superior': 40.0}))
    with pytest.raises(RuntimeError) as e:
        deepseg_models.load_crop_metadata('a_crop_model', path_model)
    assert "missing padding key" in str(e.value)

    # crop_metadata.json present, all padding keys given: returned as-is.
    full_pad = {key: float(i) for i, key in enumerate(deepseg_models.CROP_PAD_KEYS)}
    fname_json.write_text(json.dumps(full_pad))
    assert deepseg_models.load_crop_metadata('a_crop_model', path_model) == full_pad
