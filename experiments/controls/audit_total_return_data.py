"""Read original CSMAR XLSX XML streams, not the filtered three-index database."""
from pathlib import Path
import os,zipfile,re,json,csv,hashlib,xml.etree.ElementTree as ET,sqlite3
ROOT=Path(__file__).resolve().parents[2]
OUT=Path(os.environ.get('COSTER_EXPERIMENT_ROOT',str(ROOT/'runs/experiments')))/'data_audit';OUT.mkdir(parents=True,exist_ok=True)
CODES={'H00300','H00906','000300','000906','QSY301','QSY916'}
PAT=re.compile(rb'<t>(H00300|H00906|000300|000906|QSY301|QSY916)</t>')

def row_values(raw):
    elem=ET.fromstring(raw)
    return {re.sub('[0-9]','',c.attrib['r']):''.join(c.itertext()) for c in elem}

def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def main():
    files=sorted((ROOT/'datasets').rglob('IDX_Idxtrd*.xlsx'));records=[];sources=[]
    for p in files:
        count=0
        with zipfile.ZipFile(p) as z:
            sheets=[n for n in z.namelist() if re.match(r'xl/worksheets/sheet\d+\.xml$',n)]
            for sheet in sheets:
                with z.open(sheet) as f:
                    carry=b''
                    while True:
                        chunk=f.read(2*1024*1024)
                        if not chunk:break
                        data=carry+chunk;end=data.rfind(b'</row>')+6
                        if end<6:carry=data;continue
                        body=data[:end];carry=data[end:];seen=set()
                        for hit in PAT.finditer(body):
                            start=body.rfind(b'<row ',0,hit.start());stop=body.find(b'</row>',hit.end())+6
                            if start<0 or start in seen:continue
                            seen.add(start);v=row_values(body[start:stop]);code=v.get('A')
                            if code not in CODES:continue
                            records.append(dict(code=code,date=v['B'],close=float(v['F']),reported_return_pct=v.get('I'),name=v.get('J'),source=str(p.relative_to(ROOT)),sheet=sheet,row=int(re.search(rb'<row r="(\d+)"',body[start:stop])[1])))
                            count+=1
        sources.append(dict(path=str(p.relative_to(ROOT)),sha256=digest(p),target_records=count))
        (OUT/'extraction_progress.json').write_text(json.dumps(dict(files_done=len(sources),files_total=len(files),records=len(records)),indent=2),encoding='utf-8')
        print(p.parent.name,p.name,count,flush=True)
    with (OUT/'index_records_with_provenance.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    unique={};conflicts=[]
    for r in records:
        key=(r['code'],r['date'])
        if key in unique and abs(unique[key]['close']-r['close'])>1e-8:conflicts.append([unique[key],r])
        unique[key]=r
    db=ROOT/'datasets/master_ext_clean_v1/intermediate/master_ext_clean.sqlite'
    con=sqlite3.connect(db.as_uri()+'?mode=ro',uri=True)
    dates=[r[0] for r in con.execute("select trade_date from index_daily where index_code='000300' and trade_date between '2022-01-04' and '2025-12-31' order by trade_date")]
    audit={}
    for code in sorted(CODES):
        rr=sorted([v for (c,d),v in unique.items() if c==code],key=lambda r:r['date'])
        have={r['date'] for r in rr};missing=[d for d in dates if d not in have]
        audit[code]=dict(unique_records=len(rr),first=rr[0]['date'] if rr else None,last=rr[-1]['date'] if rr else None,test_dates_found=len(dates)-len(missing),missing_test_dates=missing,names=sorted({r['name'] for r in rr}))
    assert not conflicts,'Conflicting source records'
    assert len(dates)==969
    parity=[]
    for code in ['000300','000906']:
        for date,close in con.execute('select trade_date,close from index_daily where index_code=?',(code,)):
            r=unique.get((code,date))
            if r:parity.append(abs(close-r['close']))
    con.close();assert max(parity)<1e-8
    available=all(not audit[c]['missing_test_dates'] for c in ['H00300','H00906'])
    if available:
        with (OUT/'total_return_benchmarks.csv').open('w',newline='',encoding='utf-8-sig') as f:
            w=csv.DictWriter(f,fieldnames=['date','csi300','csi800']);w.writeheader()
            for date in dates:w.writerow(dict(date=int(date.replace('-','')),csi300=unique['H00300',date]['close'],csi800=unique['H00906',date]['close']))
    result=dict(status='complete',complete_test_total_return_series=available,required_dates=len(dates),sources=sources,index_summary=audit,duplicate_records=len(records)-len(unique),conflicts=conflicts,existing_price_index_max_error=max(parity),important='Existing SQLite database deliberately filters to 000300/000905/000906. Its lack of total-return codes is not evidence that raw spreadsheets lack them.')
    (OUT/'total_return_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('sources','index_summary')},ensure_ascii=False),flush=True)
    for c in ['H00300','H00906']:print(c,audit[c],flush=True)
if __name__=='__main__':main()
