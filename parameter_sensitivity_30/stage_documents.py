"""Narrow OOXML edits: two figures and their new sensitivity evidence."""
from pathlib import Path
import sys,zipfile,re,html,json,hashlib,copy
sys.stdout.reconfigure(encoding='utf-8')
W=Path('H:/RadarST_FigureRevision2_20261111');D=Path('H:/RadarST_PROJECT/NC_SUBMISSION_FINAL')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def text(p):return ''.join(html.unescape(m.group(1)) for m in re.finditer(r'<w:t\b[^>]*>(.*?)</w:t>',p,re.S))
def runs(s,marked,template=''):
 existing=[m.group() for m in re.finditer(r'<w:r(?:\s[^>]*)?>.*?</w:r>',template,re.S) if '<w:t' in m.group()];pr=re.search(r'<w:rPr\b[^>]*>.*?</w:rPr>',existing[-1],re.S) if existing else None;properties=pr.group() if pr else '<w:rPr></w:rPr>'
 if marked:
  if '<w:color' in properties:properties=re.sub(r'<w:color\b[^>]*/>','<w:color w:val="00008B"/>',properties)
  else:properties=properties.replace('</w:rPr>','<w:color w:val="00008B"/></w:rPr>')
 return '<w:r>'+properties+'<w:t xml:space="preserve">'+html.escape(s,quote=False)+'</w:t></w:r>'
def append(p,s,marked):return p[:-6]+runs(s,marked,p)+'</w:p>'
def replace(p,old,new,marked):
 value=text(p);assert value.count(old)==1,(old,value);start=value.index(old);end=start+len(old);offset=0;changes=[];first=True
 for m in re.finditer(r'(<w:t\b[^>]*>)(.*?)(</w:t>)',p,re.S):
  part=html.unescape(m.group(2));lo=max(0,start-offset);hi=min(len(part),end-offset)
  if hi>lo:
   updated=part[:lo]+(new if first else '')+part[hi:];first=False;changes.append((m.start(2),m.end(2),html.escape(updated,quote=False)))
  offset+=len(part)
 for a,b,v in reversed(changes):p=p[:a]+v+p[b:]
 # Existing marked captions/index text already use the requested dark blue;
 # explicitly mark the run that contains any newly inserted substring.
 if marked:
  def color(m):
   r=m.group()
   if new not in text(r):return r
   if '<w:color' in r:return re.sub(r'<w:color\b[^>]*/>','<w:color w:val="00008B"/>',r)
   if '</w:rPr>' in r:return r.replace('</w:rPr>','<w:color w:val="00008B"/></w:rPr>')
   return r.replace('>','><w:rPr><w:color w:val="00008B"/></w:rPr>',1)
  p=re.sub(r'<w:r(?:\s[^>]*)?>.*?</w:r>',color,p,flags=re.S)
 assert text(p)==value[:start]+new+value[end:];return p
def main():
 assert (W/'evaluation/EVALUATION_COMPLETE.json').exists()
 s=json.loads((W/'evaluation/PARAMETER_SENSITIVITY_30.json').read_text());assert len(s)==30
 gain=[v['sample_equal_size_1_10_gain'] for v in s];ming=[v['minimum_method_gain'] for v in s];allpositive=all(v['positive_samples']==12 and v['positive_methods']==14 for v in s)
 results=(' Across expression weights from 0.70 to 0.90 and accepted-area retention from 95% to 100%, '+('all 30 prespecified configurations had positive mean gains for every workflow and sample.' if allpositive else f"{sum(v['positive_methods']==14 and v['positive_samples']==12 for v in s)} of the 30 prespecified configurations had positive mean gains for every workflow and sample.")+f' Size 1–10 gain ranged from {min(gain):.4f} to {max(gain):.4f}, with minimum workflow gains from {min(ming):.4f} to {max(ming):.4f} (Fig. 5b; Supplementary Table S31).')
 methods=' We additionally evaluated all 30 combinations of five initialization expression weights (0.70, 0.75, 0.80, 0.85 and 0.90) and six accepted-area retention fractions (95%, 96%, 97%, 98%, 99% and 100%) in development. Models, candidate identities and the six-round refinement rule were unchanged. The main analysis used an expression weight of 0.80 and 97% retention (Supplementary Table S31).'
 caption=' Panel b shows the 30 combinations of expression weight and retention during initialization (Supplementary Table S31). The outlined cell marks the 0.80 weight and 97% retention. Panel d shows the 312 maps with nonzero modification. Ten maps with zero modification and zero gain lie at the origin, outside the shown range. Cohort means include all 322 maps.'
 supplement=' We evaluated the 30 combinations of expression weight and retention of accepted area during initialization, using all 168 development maps for each combination. All configurations used the same models and the six-round refinement rule. The main analyses used an expression weight of 0.80 and 97% retention.'+results
 records=[]
 for name in ['manuscript_marked.docx','Supplementary_Information.docx','Supplementary_Information_marked.docx']:
  src=D/name;original=W/'before'/name;assert sha(src)==sha(original),'User edit detected: '+name
  marked=name!='Supplementary_Information.docx';changes=[]
  with zipfile.ZipFile(src) as zi,zipfile.ZipFile(W/'staged'/name,'w',compression=zipfile.ZIP_DEFLATED) as zo:
   for info in zi.infolist():
    data=zi.read(info.filename)
    if name=='manuscript_marked.docx' and info.filename in ['word/media/image3.png','word/media/image5.png']:data=(W/'figures'/('Figure3.png' if 'image3' in info.filename else 'Figure5.png')).read_bytes()
    if info.filename=='word/document.xml':
     xml=data.decode('utf-8');count=[0]
     def edit(m):
      p=m.group();t=text(p);new=p
      if name=='manuscript_marked.docx':
       if allpositive and t=='Robust refinement of compact regions across workflows and external sections':new=replace(p,'workflows','parameter settings',marked)
       elif allpositive and t.startswith('RadarST exceeded random selection matched for both modification burden and geometry'):
        new=replace(p,'across the evaluated settings for the number of refinement rounds','across the development parameter neighborhood, across the evaluated settings for the number of refinement rounds',marked)
       elif t.startswith('Gains for compact regions stabilized over successive refinement rounds.'):new=append(p,results,marked)
       elif t.startswith('We assessed sensitivity to iteration count by evaluating complete maps'):new=append(p,methods,marked)
       elif t.startswith('Figure 3. RadarST improves'):new=replace(p,'absent exact sizes are left empty','absent exact sizes have no marker, and lines connect the available sizes',marked)
       elif t.startswith('Figure 5. Candidate selection, refinement burden and sensitivity analyses.'):new=append(p,caption,marked)
      else:
       if t.startswith('The accompanying workbook contains Supplementary Tables S1'):
        new=replace(p,'S1–S30','S1–S31',marked);new=append(new,' Supplementary Table S31 reports the 30 expression-weight and accepted-area retention combinations.',marked)
       elif t.startswith('We evaluated every stopping point from zero through six rounds without changing'):
        # Add a separate ordinary paragraph; retain the entire original one.
        pr=re.search(r'<w:pPr\b[^>]*>.*?</w:pPr>',p,re.S);new=p+'<w:p>'+(pr.group() if pr else '')+runs(supplement,marked,p)+'</w:p>'
      if new!=p:changes.append(dict(before=t,after=text(new)));count[0]+=1
      return new
     xml=re.sub(r'<w:p(?:\s[^>]*)?>.*?</w:p>',edit,xml,flags=re.S);assert count[0]==((6 if allpositive else 4) if name=='manuscript_marked.docx' else 2),(name,count);data=xml.encode('utf-8')
    zo.writestr(info,data)
  with zipfile.ZipFile(src) as zi,zipfile.ZipFile(W/'staged'/name) as zo:
   changed=[n for n in zi.namelist() if zi.read(n)!=zo.read(n)];assert set(changed)==({'word/document.xml','word/media/image3.png','word/media/image5.png'} if name=='manuscript_marked.docx' else {'word/document.xml'});assert zo.testzip() is None
  records.append(dict(document=name,original_sha256=sha(src),staged_sha256=sha(W/'staged'/name),changed_entries=changed,text_changes=changes))
 (W/'qa/DOCUMENT_EDIT_AUDIT.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8');print('THREE_DOCUMENTS_STAGED')
if __name__=='__main__':main()
