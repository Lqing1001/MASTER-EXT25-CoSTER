"""Re-evaluate unchanged submitted long-only returns against raw CSMAR TR indices."""
from common import *
import pandas as pd

def benchmark_returns():
    p=OUT/'data_audit/total_return_benchmarks.csv'
    a=json.loads((p.parent/'total_return_audit.json').read_text(encoding='utf-8'))
    assert a['complete_test_total_return_series'] and not a['conflicts']
    tr=pd.read_csv(p).set_index('date').sort_index();assert len(tr)==969 and tr.index.is_unique and (tr>0).all().all()
    returns=tr.pct_change().iloc[1:];assert returns.notna().all().all()
    return returns

def main():
    setup();r=benchmark_returns();d=OUT/'data_audit'
    source=OLD/'final_paper_evidence/longonly_daily.csv';df=pd.read_csv(source);rows=[];bench=[]
    for u in UNIVERSES:
        br=r[u];cagr=float(np.prod(1+br)**(252/len(br))-1)
        bench.append(dict(universe=u,index_code='H00300' if u=='csi300' else 'H00906',return_dates=len(br),CAGR=cagr))
        for (model,seed),g in df[df.universe==u].groupby(['model','seed']):
            g=g.set_index('date').sort_index();assert g.index.equals(br.index)
            active=g.net-br
            rows.append(dict(universe=u,model=model,seed=int(seed),net_CAGR=float(np.prod(1+g.net)**(252/len(g))-1),TR_index_CAGR=cagr,net_excess_annual=float(252*active.mean()),net_IR=float(np.sqrt(252)*active.mean()/active.std(ddof=1))))
    csvout(d/'total_return_index_summary.csv',bench);csvout(d/'submitted_longonly_TR_per_seed.csv',rows)
    out=pd.DataFrame(rows).groupby(['universe','model'])[['net_CAGR','TR_index_CAGR','net_excess_annual','net_IR']].agg(['mean','std']);out.to_csv(d/'submitted_longonly_TR_summary.csv',encoding='utf-8-sig')
    dump(d/'benchmark_integration.json',dict(status='verified_and_recomputed_submitted_three_seeds',benchmark_file=str(d/'total_return_benchmarks.csv'),benchmark_sha256=sha(d/'total_return_benchmarks.csv'),strategy_source=str(source),strategy_sha256=sha(source),no_strategy_return_changes=True,dates=968,stock_return_convention='original cumulative-backward-adjusted prices retained; exact corporate-action reinvestment convention is not redefined by this benchmark substitution'))
    print(out.to_string(),flush=True)
if __name__=='__main__':main()
