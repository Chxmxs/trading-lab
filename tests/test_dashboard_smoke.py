# encoding: utf-8
from companion.explorer.dashboard import build_layout
def test_dashboard_smoke(tmp_path):
    (tmp_path/"artifacts"/"feature_rank"/"20250101T000000Z").mkdir(parents=True, exist_ok=True)
    (tmp_path/"artifacts"/"feature_rank"/"20250101T000000Z"/"importance_rank.csv").write_text("symbol,timeframe,feature,mean_importance,runs,rank\nBTCUSD,15m,asopr,0.7,3,1\n",encoding="utf-8")
    layout = build_layout(str(tmp_path))
    assert layout is not None
