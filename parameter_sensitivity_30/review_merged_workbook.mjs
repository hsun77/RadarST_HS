import fs from 'node:fs/promises';
import { FileBlob, SpreadsheetFile } from '@oai/artifact-tool';
const w = 'H:/RadarST_FigureRevision2_20261111';
const file = w + '/staged/Supplementary_Tables.xlsx';
const wb = await SpreadsheetFile.importXlsx(await FileBlob.load(file));
const source = JSON.parse(await fs.readFile(w + '/evaluation/PARAMETER_SENSITIVITY_30.json', 'utf8'));
const keys = Object.keys(source[0]);
const col = n => {let s='';for(n++;n>0;n=Math.floor((n-1)/26))s=String.fromCharCode(65+(n-1)%26)+s;return s;};
const sheet = wb.worksheets.getItem('S31');
const values = sheet.getRangeByIndexes(0,0,31,keys.length).values;
if (JSON.stringify(values[0]) !== JSON.stringify(keys)) throw Error('S31 headers differ');
for (let i=0;i<30;i++) for(let j=0;j<keys.length;j++) {
 const expected=source[i][keys[j]],actual=values[i+1][j];
 if (typeof expected==='number' ? Math.abs(actual-expected)>1e-12 : actual!==expected) throw Error(`S31 mismatch ${i},${j}`);
}
const rendered=[];
const full=await wb.render({sheetName:'S31',range:`A1:${col(keys.length-1)}31`,scale:1,format:'png'});
await fs.writeFile(w+'/qa/S31_merged_full.png',new Uint8Array(await full.arrayBuffer()));
for (let start=0;start<keys.length;start+=8) {
 const end=Math.min(start+7,keys.length-1);
 const range=`${col(start)}1:${col(end)}12`;
 const png=await wb.render({sheetName:'S31',range,scale:1,format:'png'});
 const path=w+`/qa/S31_merged_${col(start)}_${col(end)}.png`;
 await fs.writeFile(path,new Uint8Array(await png.arrayBuffer()));rendered.push({range,path});
}
const preview=await wb.render({sheetName:'Index',range:'A27:E32',scale:1,format:'png'});
await fs.writeFile(w+'/qa/S31_index_added.png',new Uint8Array(await preview.arrayBuffer()));
await fs.writeFile(w+'/qa/MERGED_WORKBOOK_REIMPORT_AUDIT.json',JSON.stringify({status:'PASS',file,rows:30,columns:keys.length,all_values_equal_full_precision_summary:true,rendered},null,2));
console.log(JSON.stringify({status:'PASS',rows:30,columns:keys.length,rendered}));
