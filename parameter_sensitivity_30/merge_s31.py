"""Package Artifact-authored S31, leaving all old table XML byte-identical."""
from pathlib import Path
import zipfile,json,hashlib,copy,re
from lxml import etree as E
W=Path('H:/RadarST_FigureRevision2_20261111');D=Path('H:/RadarST_PROJECT/NC_SUBMISSION_FINAL');S='http://schemas.openxmlformats.org/spreadsheetml/2006/main';R='http://schemas.openxmlformats.org/officeDocument/2006/relationships';P='http://schemas.openxmlformats.org/package/2006/relationships';C='http://schemas.openxmlformats.org/package/2006/content-types'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def emit(r):return E.tostring(r,encoding='utf-8',xml_declaration=True)
def main():
 source=W/'before/Supplementary_Tables.xlsx';assert sha(source)==sha(D/source.name)
 with zipfile.ZipFile(source) as z:old={n:z.read(n) for n in z.namelist()}
 with zipfile.ZipFile(W/'staged/S31_authored.xlsx') as z:new={n:z.read(n) for n in z.namelist()}
 sheet=E.fromstring(new['xl/worksheets/sheet1.xml'])
 for view in sheet.findall('.//{'+S+'}sheetView'):view.attrib.pop('tabSelected',None)
 styles=E.fromstring(old['xl/styles.xml']);newstyles=E.fromstring(new['xl/styles.xml']);xfs=styles.find('{'+S+'}cellXfs');nxf=newstyles.find('{'+S+'}cellXfs');formats=styles.find('{'+S+'}numFmts')
 if formats is None:formats=E.Element('{'+S+'}numFmts',count='0');styles.insert(0,formats)
 numids={int(x.get('numFmtId')):x.get('formatCode') for x in formats};maxnum=max([163]+list(numids));newnum={int(x.get('numFmtId')):x.get('formatCode') for x in newstyles.find('{'+S+'}numFmts')} if newstyles.find('{'+S+'}numFmts') is not None else {};stylemap={}
 for c in sheet.findall('.//{'+S+'}c'):
  is_header=bool(re.search(r'[^0-9]1$',c.get('r')));sid=int(c.get('s','0'));num=int(nxf[sid].get('numFmtId','0'));key=(is_header,num)
  if key not in stylemap:
   if is_header:stylemap[key]=74
   elif not num:stylemap[key]=63
   else:
    destnum=num
    if num>=164:
     code=newnum[num];destnum=next((k for k,v in numids.items() if v==code),None)
     if destnum is None:maxnum+=1;destnum=maxnum;E.SubElement(formats,'{'+S+'}numFmt',numFmtId=str(destnum),formatCode=code);numids[destnum]=code
    xf=copy.deepcopy(xfs[63]);xf.set('numFmtId',str(destnum));stylemap[key]=len(xfs);xfs.append(xf)
  c.set('s',str(stylemap[key]))
 # Only the new sheet's strings need conversion; old shared strings remain untouched.
 shared=[''.join(t.itertext()) for t in E.fromstring(new['xl/sharedStrings.xml'])] if 'xl/sharedStrings.xml' in new else []
 for c in sheet.findall('.//{'+S+'}c'):
  if c.get('t')=='s':
   v=c.find('{'+S+'}v');value=shared[int(v.text)];c.remove(v);c.set('t','inlineStr');E.SubElement(E.SubElement(c,'{'+S+'}is'),'{'+S+'}t').text=value
 xfs.set('count',str(len(xfs)));formats.set('count',str(len(formats)))
 book=E.fromstring(old['xl/workbook.xml']);sheets=book.find('{'+S+'}sheets');sid=max(int(x.get('sheetId')) for x in sheets)+1;assert not any(x.get('name')=='S31' for x in sheets)
 rels=E.fromstring(old['xl/_rels/workbook.xml.rels']);used={x.get('Id') for x in rels};rid='rId'+str(max([0]+[int(v[3:]) for v in used if v.startswith('rId') and v[3:].isdigit()])+1);target=f'worksheets/sheet{sid}.xml'
 E.SubElement(sheets,'{'+S+'}sheet',{'name':'S31','sheetId':str(sid),'{'+R+'}id':rid});E.SubElement(rels,'{'+P+'}Relationship',Id=rid,Type=R+'/worksheet',Target=target)
 index=E.fromstring(old['xl/worksheets/sheet1.xml']);data=index.find('{'+S+'}sheetData');row=E.SubElement(data,'{'+S+'}row',r='32',ht='46',customHeight='1');values=['S31','Development sensitivity across 30 expression-weight and accepted-area retention combinations',30,len(json.loads((W/'evaluation/PARAMETER_SENSITIVITY_30.json').read_text())[0]),'All 168 maps per configuration; frozen models and six refinement rounds; reported configuration retained.']
 for col,value in zip('ABCDE',values):
  c=E.SubElement(row,'{'+S+'}c',r=col+'32',s='63')
  if isinstance(value,str):c.set('t','inlineStr');E.SubElement(E.SubElement(c,'{'+S+'}is'),'{'+S+'}t').text=value
  else:E.SubElement(c,'{'+S+'}v').text=str(value)
 if index.find('{'+S+'}dimension') is not None:index.find('{'+S+'}dimension').set('ref','A1:E32')
 types=E.fromstring(old['[Content_Types].xml']);E.SubElement(types,'{'+C+'}Override',PartName='/xl/'+target,ContentType='application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml')
 changes={'xl/'+target:emit(sheet),'xl/workbook.xml':emit(book),'xl/_rels/workbook.xml.rels':emit(rels),'xl/styles.xml':emit(styles),'xl/worksheets/sheet1.xml':emit(index),'[Content_Types].xml':emit(types)}
 # Spreadsheet application properties are descriptive, but keep their sheet count accurate.
 if 'docProps/app.xml' in old:
  app=E.fromstring(old['docProps/app.xml']);vt='http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes';ep='http://schemas.openxmlformats.org/officeDocument/2006/extended-properties';titles=app.find('{'+ep+'}TitlesOfParts/{'+vt+'}vector')
  if titles is not None:E.SubElement(titles,'{'+vt+'}lpstr').text='S31';titles.set('size',str(len(titles)))
  pairs=app.find('{'+ep+'}HeadingPairs/{'+vt+'}vector')
  if pairs is not None:
   variants=list(pairs)
   for i,v in enumerate(variants[:-1]):
    label=v.find('{'+vt+'}lpstr')
    if label is not None and label.text=='Worksheets':count=variants[i+1].find('{'+vt+'}i4');count.text=str(int(count.text)+1)
  changes['docProps/app.xml']=emit(app)
 out=W/'staged/Supplementary_Tables.xlsx'
 with zipfile.ZipFile(out,'w',compression=zipfile.ZIP_DEFLATED) as z:
  for n in old:z.writestr(n,changes.get(n,old[n]))
  z.writestr('xl/'+target,changes['xl/'+target])
 with zipfile.ZipFile(out) as z:
  assert z.testzip() is None
  for n,v in old.items():
   if n not in changes:assert z.read(n)==v
 # Strictly compare the complete pre-existing index rows and pre-existing styles.
 before=E.fromstring(old['xl/worksheets/sheet1.xml']).find('{'+S+'}sheetData');assert [emit(x) for x in list(data)[:-1]]==[emit(x) for x in before]
 assert [emit(x) for x in list(xfs)[:len(E.fromstring(old['xl/styles.xml']).find('{'+S+'}cellXfs'))]]==[emit(x) for x in E.fromstring(old['xl/styles.xml']).find('{'+S+'}cellXfs')]
 report=dict(status='PASS',source_sha256=sha(source),staged_sha256=sha(out),new_sheet='S31',artifact_authored=True,all_S1_through_S30_worksheet_XML_byte_identical=True,existing_index_rows_preserved=True,existing_cell_styles_preserved=True,changed_package_entries=list(changes))
 (W/'qa/WORKBOOK_APPEND_AUDIT.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report))
if __name__=='__main__':main()
