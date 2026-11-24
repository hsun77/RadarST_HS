import fs from 'node:fs/promises';
import { FileBlob, SpreadsheetFile, Workbook } from '@oai/artifact-tool';
const w='H:/RadarST_FigureRevision2_20261111';
if(process.argv.includes('--inspect-before')){
 const wb=await SpreadsheetFile.importXlsx(await FileBlob.load(w+'/before/Supplementary_Tables.xlsx'));
 console.log((await wb.inspect({kind:'region',sheetId:'S30',range:'A1:I7',maxChars:2200})).ndjson);
 const preview=await wb.render({sheetName:'S30',range:'A1:F7',scale:1,format:'png'});
 await fs.writeFile(w+'/qa/S30_before.png',new Uint8Array(await preview.arrayBuffer()));
}else{
 const rows=JSON.parse(await fs.readFile(w+'/evaluation/PARAMETER_SENSITIVITY_30.json','utf8'));
 const cols=Object.keys(rows[0]);
 const wb=Workbook.create();const s=wb.worksheets.add('S31');
 s.getRangeByIndexes(0,0,rows.length+1,cols.length).values=[cols,...rows.map(r=>cols.map(k=>r[k]??null))];
 const all=s.getRangeByIndexes(0,0,rows.length+1,cols.length);
 all.format.font={name:'Arial',size:10};all.format.wrapText=true;
 all.format.verticalAlignment='center';all.format.columnWidth=19;all.format.rowHeight=20;
 s.getRangeByIndexes(0,0,1,cols.length).format.font={name:'Arial',size:10,bold:true};
 s.getRangeByIndexes(0,0,1,cols.length).format.rowHeight=52;
 s.getRange('A:A').format.columnWidth=16;
 s.getRange('I:I').format.columnWidth=28;
 s.getRange('J:J').format.columnWidth=22;
 s.getRange('T:T').format.columnWidth=26;
 s.showGridLines=false;s.freezePanes.freezeRows(1);
 for(let j=1;j<cols.length;j++){
  const key=cols[j];const r=s.getRangeByIndexes(1,j,rows.length,1);
  if(['expression_weight','expected_gain_weight','retention_fraction'].includes(key))r.setNumberFormat('0.00');
  else if(['exact','near_0_8','modified_fraction','loss_event_fraction','size1_2_exact','size3_5_exact','size6_10_exact'].includes(key))r.setNumberFormat('0.0000%');
  else if(!['formal_default','sections','workflows','maps','positive_methods','positive_samples'].includes(key))r.setNumberFormat('0.000000');
 }
 wb.recalculate();
 console.log((await wb.inspect({kind:'region',sheetId:'S31',range:'A1:N5',maxChars:2200,tableMaxRows:5,tableMaxCols:14})).ndjson);
 const preview=await wb.render({sheetName:'S31',range:'A1:N12',scale:1,format:'png'});
 await fs.writeFile(w+'/qa/S31_preview.png',new Uint8Array(await preview.arrayBuffer()));
 await (await SpreadsheetFile.exportXlsx(wb)).save(w+'/staged/S31_authored.xlsx');
}
