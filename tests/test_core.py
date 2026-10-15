import json
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest


def test_fixed_config_rejects_parameter_changes(tmp_path):
    from radarst.config import read_config
    p=tmp_path/'bad.yaml';p.write_text('retention: 0.90\n')
    with pytest.raises(ValueError,match='retention'): read_config(p)


def test_missing_model_is_not_a_random_model(capsys,tmp_path):
    from radarst.cli import main
    out=tmp_path/'new'
    with pytest.raises(SystemExit) as e:
        main(['infer','--expression','missing.h5','--coordinates','missing.csv','--upstream-labels','missing_labels.csv','--dataset','synthetic','--workflow','test','--output',str(out)])
    assert e.value.code==2 and not out.exists()
    assert 'No weights are bundled' in capsys.readouterr().err


def test_synthetic_data_are_explicit(tmp_path):
    from radarst.synthetic import generate
    path=tmp_path/'synthetic';generate(path)
    assert 'artificial' in (path/'SYNTHETIC.txt').read_text()
    assert len(pd.read_csv(path/'coordinates.csv'))==81


def test_component_best_iou_and_connectivity():
    from radarst.engine import _load
    f=_load('metrics')
    ids=['a','b','c','d']
    adj=[[1],[0,2],[1,3],[2]]
    regions,_,_=f.label_connected_components(pd.Series(['x','y','x','x'],index=ids),ids,adj)
    assert sorted(map(len,regions.values()))==[1,1,2]
    match=f.true_best_match_from_regions(regions,{'c','d'})
    assert match['iou']==1 and match['overlap']==2


def test_atomic_retention_does_not_split_candidates():
    from radarst.engine import _load
    f=_load('retention')
    frame=pd.DataFrame({'n_spots':[3,2],'selection_score':[2.,1.],'candidate_key':['a','b'],'member_spots':['a;b;c','d;e']})
    assignment,n,used,budget=f.retained_assignment(frame,.97,'synthetic','test')
    assert (n,used,budget)==(1,3,4)
    assert set(assignment)=={'a','b','c'}


def test_groupwise_rank_target_is_original_definition():
    from radarst.engine import training_functions
    f=training_functions()
    frame=pd.DataFrame({'dataset_id':['s']*4,'method':['w']*4})
    result=f.rank.group_rank_target(frame,np.array([0,.2,.2,-1]))
    np.testing.assert_array_equal(result,[0,.875,.875,0])


def test_loso_loader_never_reads_excluded_table(tmp_path):
    from radarst.training import load_fold
    from radarst.frontend import sha
    a=tmp_path/'included.parquet'
    pd.DataFrame({'dataset_id':['included'],'method':['map']}).to_parquet(a)
    records=[dict(dataset='included',workflow='map',table=a.name,sha256=sha(a)),dict(dataset='heldout',workflow='map',table='DOES_NOT_EXIST.parquet',sha256='not_read')]
    result=load_fold(records,tmp_path,{'included'})
    assert len(result)==1 and result[0].dataset_id.iloc[0]=='included'


def test_training_without_teacher_is_explicitly_blocked(tmp_path):
    from radarst.training import prepare
    p=tmp_path/'manifest.yaml'
    p.write_text('samples:\n  - dataset: synthetic\n    cohort: development\n')
    with pytest.raises(ValueError,match='teacher_acceptance'):
        prepare(p,tmp_path/'output')
    assert not (tmp_path/'output').exists()


def test_external_training_is_rejected(tmp_path):
    from radarst.training import prepare
    p=tmp_path/'manifest.yaml'
    p.write_text('samples:\n  - dataset: synthetic\n    cohort: external\n')
    with pytest.raises(ValueError,match='development'):
        prepare(p,tmp_path/'output')


def test_canonical_membership_is_not_a_short_id():
    from radarst.engine import functions
    f=functions()
    assert f.i.canonical_spots('c;a;c;b')=='a;b;c'
    assert f.i.candidate_key('sample','singleton','a')=='sample||singleton||a'
