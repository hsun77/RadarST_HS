"""Legal artificial inputs; no distributed experimental data or weights."""
import json
import numpy as np
import pandas as pd
import pytest


def test_exclusion_precedes_file_checks(tmp_path):
    import yaml
    from radarst.teacher import selected_samples
    expression=tmp_path/'counts.h5'; expression.touch()
    coordinates=tmp_path/'coordinates.csv'; coordinates.touch()
    manifest=tmp_path/'manifest.yaml'
    manifest.write_text(yaml.safe_dump(dict(samples=[
        dict(dataset='included',cohort='development',expression=expression.name,coordinates=coordinates.name),
        dict(dataset='heldout',cohort='development',expression='NO_READ.h5',coordinates='NO_READ.csv',reference='NO_READ_truth.csv')
    ])))
    _,included=selected_samples(manifest,'heldout')
    assert [s['dataset'] for s in included]==['included']


def test_teacher_requires_actual_multiworkflow_inputs(tmp_path):
    from radarst.teacher import generate_one
    with pytest.raises(ValueError,match='14 distinct'):
        generate_one(dict(upstream=[dict(workflow='only_one')]),tmp_path,tmp_path/'out')
    assert not (tmp_path/'out').exists()


def test_original_short_id_join_is_not_silently_deduplicated(monkeypatch):
    from radarst.engine import _load
    f=_load('teacher_sensor',patch_path_text=str)
    # An artificial collision tests the original join, not historical recovery.
    candidates=pd.DataFrame(dict(sample_id=['toy']*2,candidate_id=['same']*2,member_spots=['a','b']))
    canonical=pd.DataFrame(dict(sample_id=['toy']*3,invalid_coordinate_spot=False,
        spot_id_normalized=['a','b','c'],x=[0,1,2],y=[0,0,0]))
    monkeypatch.setitem(f.add_sensor_features.__globals__,'build_sensor_matrix',
        lambda *args:(np.zeros((3,14),dtype=int),list(range(14))))
    actual=f.add_sensor_features(candidates,canonical,pd.DataFrame())
    assert len(actual)==4
    assert actual.member_spots.tolist()==['a','a','b','b']


def test_teacher_does_not_read_forbidden_supervised_columns(tmp_path):
    from radarst.engine import _load,functions
    import radarst.teacher
    from pathlib import Path
    config=json.loads(Path(radarst.teacher.__file__).with_name('teacher_protocol.json').read_text())['evidence']
    frame=pd.DataFrame(dict(dataset_id=['toy']*3,sample_id=['toy']*3,candidate_id=['a','b','c'],
        proposal_type=['singleton']*3,member_spots=['a','b','c'],n_spots=[1]*3,
        proposal_score=[1.,2.,3.],sensor_boundary_mean=[.1,.8,.9],
        sensor_island_support_frac=[0.,.5,1.],sensor_rare_support_frac=[0.,.5,1.],
        sensor_loso_predicted_IoU=[100.,-10.,50.],training_best_IoU=[1.,0.,1.]))
    path=tmp_path/'new.csv';frame.to_csv(path,index=False)
    config['candidate_input']['path']=str(path)
    f=_load('teacher_evidence',B67=functions().b67)
    one=f.prepare_candidate_evidence(config)
    frame['sensor_loso_predicted_IoU']=-999;frame['training_best_IoU']=999
    frame.to_csv(path,index=False)
    two=f.prepare_candidate_evidence(config)
    pd.testing.assert_frame_equal(one,two)
    assert 'training_best_IoU' not in one


def test_training_preparation_excludes_missing_heldout_truth(tmp_path,monkeypatch):
    import yaml
    from radarst.training import prepare
    monkeypatch.setenv('PYTHONHASHSEED','0')
    manifest=tmp_path/'manifest.yaml'
    manifest.write_text(yaml.safe_dump(dict(samples=[
        dict(dataset='included',cohort='development'),
        dict(dataset='heldout',cohort='development',reference='DO_NOT_READ.csv')
    ])))
    with pytest.raises(ValueError,match='teacher_acceptance'):
        prepare(manifest,tmp_path/'out',held_out='heldout')
    assert not (tmp_path/'out').exists()
