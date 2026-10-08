#!/usr/bin/env python3
"""Build a portable, offline CUT visual QC gallery using only the standard library."""
import argparse
import csv
import hashlib
import html
import json
from pathlib import Path
import re
import shutil

ROLES = ('fake_B', 'real_A', 'real_B', 'idt_B')
EXTENSIONS = {'.png', '.jpg', '.jpeg', '.webp', '.bmp', '.gif'}
SUFFIX = re.compile(r'^(.+)_(fake_B|real_A|real_B|idt_B)$')


def extract_basename(path):
    match = SUFFIX.fullmatch(path.stem)
    if match:
        return match.groups()
    if path.parent.name in ROLES:
        return path.stem, path.parent.name
    return None


def find_result_images(results_dir, recursive=False):
    # Role folders are a standard CUT layout and do not require --recursive.
    paths = results_dir.rglob('*') if recursive else (
        p for folder in [results_dir] + [results_dir / r for r in ROLES]
        if folder.is_dir() for p in folder.iterdir())
    return sorted(p for p in paths if p.is_file() and p.suffix.lower() in EXTENSIONS
                  and extract_basename(p))


def group_images(paths):
    groups = {}
    for path in paths:
        basename, role = extract_basename(path)
        group = groups.setdefault(basename, {})
        if role in group:
            raise ValueError(f'Ambiguous {basename}/{role}: {group[role]} and {path}')
        group[role] = path
    return groups


def parse_metadata(basename):
    match = re.fullmatch(r'subject(?P<subject>\d+)_(?P<eye>OS|OD)_bscan_(?P<bscan>\d+)', basename, re.I)
    return match.groupdict() if match else dict(subject='', eye='', bscan='')


def natural_key(value):
    return tuple((1, int(part)) if part.isdigit() else (0, part.casefold())
                 for part in re.split(r'(\d+)', value))


def sort_key(basename):
    meta = parse_metadata(basename)
    if meta['subject']:
        return (0, int(meta['subject']), meta['eye'].upper(), int(meta['bscan']), basename)
    return (1, natural_key(basename), basename)


def write_manifest(path, rows):
    fields = ['basename'] + [r + '_path' for r in ROLES] + ['complete_group', 'subject', 'eye', 'bscan']
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in fields})


def write_missing_report(path, rows):
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=['basename', 'missing_role', 'expected_file'])
        writer.writeheader()
        for row in rows:
            for role in ROLES:
                if not row[role + '_path']:
                    writer.writerow(dict(basename=row['basename'], missing_role=role,
                                         expected_file=f"{row['basename']}_{role}.<extension> or {role}/{row['basename']}.<extension>"))


def build_html(rows, title, dataset_id):
    payload = json.dumps(dict(cases=rows, datasetId=dataset_id), ensure_ascii=True).replace('<', '\\u003c')
    return TEMPLATE.replace('__TITLE__', html.escape(title)).replace('__DATA__', payload)


TEMPLATE = '''<!doctype html>
<html lang="de"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
*{box-sizing:border-box}body{margin:0;background:#14171b;color:#edf0f3;font:16px system-ui,sans-serif}main{padding:24px;max-width:1900px;margin:auto}h1{font-size:24px}h2{font-size:19px;overflow-wrap:anywhere}button,input{font:inherit;padding:8px 12px;border:1px solid #727d89;border-radius:5px;background:#252c34;color:inherit}button{cursor:pointer}button:disabled{opacity:.4;cursor:default}button:focus-visible,input:focus-visible{outline:3px solid #64c7ef}nav{display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin:18px 0}#position{margin-left:auto}input{width:90px}.panels{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:14px}.panel{margin:0;background:#1d2229;border:1px solid #414b57;border-radius:6px;overflow:hidden}.panel h3{margin:12px}.frame{height:clamp(220px,32vw,480px);display:flex;align-items:center;justify-content:center;background:#08090a}.frame button{width:100%;height:100%;padding:0;border:0;background:transparent}.frame img{width:100%;height:100%;object-fit:contain}.missing{color:#ffbf75}.qc{display:flex;align-items:center;flex-wrap:wrap;gap:8px;margin:14px 0}.qc span{min-width:290px}.qc button[aria-pressed=true]{outline:3px solid #8cd7ff;background:#35536c}details{background:#20262d;padding:14px;border-radius:6px;line-height:1.6}#storage{color:#ffbf75}dialog{max-width:96vw;max-height:96vh;background:#14171b;color:white;border:1px solid #73808d;padding:16px}dialog::backdrop{background:#000c}.zoomviewport{overflow:auto;max-width:90vw;max-height:78vh;margin-top:12px}#zoomImage{display:block;max-width:none}#zoomImage.fit{max-width:88vw;max-height:76vh;width:auto;height:auto}small{color:#bdc7d2}@media(max-width:900px){.panels{grid-template-columns:repeat(2,minmax(0,1fr))}.frame{height:300px}}
</style><main><h1>__TITLE__</h1>
<details open><summary>Interpretation · unpaired CUT</summary>
<b>real_A</b>: Originalbild aus Domain A. <b>fake_B</b>: Translation von real_A nach Domain B.<br>
<b>real_B</b>: echtes Domain-B-Bild; bei unpaired Training <b>keine Ground Truth für fake_B</b>. Es wird keine pixelweise Übereinstimmung erwartet.<br>
<b>idt_B</b>: Identity-Ausgabe eines Domain-B-Bildes.<br>
Bleibt die Anatomie von real_A → fake_B erhalten, während sich der Scannerstil Richtung Domain B verändert? Verändert real_B → idt_B das Bild nur minimal?
</details><h2 id="caseTitle"></h2><small id="metadata"></small>
<nav><button id="first">First</button><button id="prev">Previous</button><button id="next">Next</button><button id="last">Last</button><label>Fall <input id="jump" type="number" min="1" step="1"></label><span id="position" aria-live="polite"></span></nav>
<div class="panels" id="panels"></div><div id="ratings"></div>
<button id="export">Export QC ratings (JSON)</button><p id="storage" role="status"></p><small>Originaldateien unverändert. Klick: Zoom; Pfeiltasten: Navigation; Escape: Zoom schließen. Bewertungen werden nach Möglichkeit lokal in diesem Browser gespeichert. Für eine dauerhafte Sicherung JSON exportieren.</small>
<dialog id="zoom"><div><b id="zoomTitle"></b> <button id="fit">Fit</button> <button id="original">100% (Originalpixel)</button> <button id="close">Schließen</button></div><div class="zoomviewport"><img id="zoomImage" alt=""></div></dialog>
</main><script id="data" type="application/json">__DATA__</script><script>
'use strict';
const {cases,datasetId}=JSON.parse(document.getElementById('data').textContent);
const roles=['fake_B','real_A','real_B','idt_B'];
const criteria=[['overall','Overall QC'],['anatomy','Anatomy preservation · real_A → fake_B'],['appearance','Target-domain appearance · unpaired'],['identity','Identity preservation · real_B → idt_B']];
const $=id=>document.getElementById(id), key='cut-qc-v1:'+datasetId;
let index=0,ratings=Object.create(null);
try{const saved=JSON.parse(localStorage.getItem(key)||'{}');if(saved && typeof saved==='object' && !Array.isArray(saved))ratings=Object.assign(Object.create(null),saved);}catch(e){$('storage').textContent='Lokales Speichern nicht verfügbar oder gespeicherte Daten nicht lesbar. Bitte JSON exportieren.';}
function save(){try{localStorage.setItem(key,JSON.stringify(ratings));}catch(e){$('storage').textContent='Bewertungen nur im Arbeitsspeicher! Bitte vor dem Schließen JSON exportieren.';}}
function renderRatings(){
 $('ratings').replaceChildren(); if(!cases.length)return;
 for(const [criterion,label] of criteria){const row=document.createElement('div');row.className='qc';const text=document.createElement('span');text.textContent=label;row.append(text);
 for(const value of ['PASS','SUSPICIOUS','FAIL']){const b=document.createElement('button');b.textContent=value;b.setAttribute('aria-pressed',String(ratings[cases[index].basename]?.[criterion]===value));b.onclick=()=>{const name=cases[index].basename;const entry=ratings[name]||{};if(entry[criterion]===value)delete entry[criterion];else entry[criterion]=value;entry.updated_at=new Date().toISOString();ratings[name]=entry;save();renderRatings();};row.append(b);} $('ratings').append(row);}
}
function render(){
 $('first').disabled=$('prev').disabled=!cases.length||index===0;
 $('next').disabled=$('last').disabled=!cases.length||index===cases.length-1;
 $('jump').disabled=!cases.length;$('jump').max=cases.length;$('jump').value=cases.length?index+1:'';
 $('position').textContent=cases.length?`${index+1} / ${cases.length}`:'0 / 0';$('panels').replaceChildren();
 if(!cases.length){$('caseTitle').textContent='Keine passenden Ergebnisbilder gefunden.';return;}
 const c=cases[index];$('caseTitle').textContent=c.basename;$('metadata').textContent=c.subject?`Subject ${c.subject} | ${c.eye} | B-Scan ${c.bscan}`:'';
 for(const role of roles){const panel=document.createElement('section');panel.className='panel';const h=document.createElement('h3');h.textContent=role;const frame=document.createElement('div');frame.className='frame';panel.append(h,frame);
 if(c.images[role]){const b=document.createElement('button');b.setAttribute('aria-label',role+' vergrößern');const img=document.createElement('img');img.alt=role+' · '+c.basename;img.src=c.images[role];img.onerror=()=>{frame.textContent='missing / Bild nicht ladbar';frame.classList.add('missing');};b.append(img);b.onclick=()=>{$('zoomTitle').textContent=img.alt;$('zoomImage').src=img.src;$('zoomImage').alt=img.alt;$('zoomImage').className='fit';$('zoom').showModal();};frame.append(b);}else{frame.textContent='missing';frame.classList.add('missing');} $('panels').append(panel);}
 renderRatings();
}
function go(n){index=Math.max(0,Math.min(cases.length-1,n));render();}
$('first').onclick=()=>go(0);$('prev').onclick=()=>go(index-1);$('next').onclick=()=>go(index+1);$('last').onclick=()=>go(cases.length-1);
$('jump').onchange=()=>{const n=Number($('jump').value);if(Number.isInteger(n)&&n>=1&&n<=cases.length)go(n-1);else $('jump').value=cases.length?index+1:'';};
document.addEventListener('keydown',e=>{if($('zoom').open||['INPUT','TEXTAREA','SELECT'].includes(e.target.tagName))return;if(e.key==='ArrowLeft'){e.preventDefault();go(index-1);}if(e.key==='ArrowRight'){e.preventDefault();go(index+1);}});
$('close').onclick=()=>$('zoom').close();$('zoom').onclick=e=>{if(e.target===$('zoom')){const r=$('zoom').getBoundingClientRect();if(e.clientX<r.left||e.clientX>r.right||e.clientY<r.top||e.clientY>r.bottom)$('zoom').close();}};
$('fit').onclick=()=>$('zoomImage').className='fit';$('original').onclick=()=>$('zoomImage').className='';
$('export').onclick=()=>{const data={schema_version:1,dataset_id:datasetId,exported_at:new Date().toISOString(),interpretation:'Unpaired: real_B is not ground truth for fake_B.',cases:cases.map(c=>({basename:c.basename,subject:c.subject,eye:c.eye,bscan:c.bscan,complete_group:c.complete_group,missing_roles:roles.filter(r=>!c.images[r]),ratings:ratings[c.basename]||{}}))};const url=URL.createObjectURL(new Blob([JSON.stringify(data,null,2)],{type:'application/json'}));const a=document.createElement('a');a.href=url;a.download='cut_qc_ratings.json';document.body.append(a);a.click();a.remove();setTimeout(()=>URL.revokeObjectURL(url),1000);};
render();
</script></html>'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--title', default='CUT · Visual QC')
    parser.add_argument('--recursive', action='store_true')
    args = parser.parse_args()
    source, output = args.results_dir.resolve(), args.output_dir.resolve()
    if not source.is_dir():
        parser.error(f'Results directory does not exist: {source}')
    if source == output or source in output.parents or output in source.parents:
        parser.error('Results and output directories must be separate, non-nested directories.')
    try:
        groups = group_images(find_result_images(source, args.recursive))
        output.mkdir(parents=True, exist_ok=True)
        assets = output / 'assets'
        assets.mkdir(exist_ok=True)
        rows = []
        for basename in sorted(groups, key=sort_key):
            group = groups[basename]
            row = dict(basename=basename, complete_group=all(r in group for r in ROLES),
                       **parse_metadata(basename), images={})
            for role in ROLES:
                path = group.get(role)
                row[role + '_path'] = str(path) if path else ''
                if path:
                    # Deterministic safe URLs; copy bytes without decoding/resampling.
                    name = hashlib.sha256(str(path.relative_to(source)).encode()).hexdigest() + path.suffix.lower()
                    shutil.copyfile(path, assets / name)
                    row['images'][role] = 'assets/' + name
            rows.append(row)
        write_manifest(output / 'viewer_manifest.csv', rows)
        write_missing_report(output / 'missing_files.csv', rows)
        dataset_id = hashlib.sha256(str(source).encode()).hexdigest()
        (output / 'index.html').write_text(build_html(rows, args.title, dataset_id), encoding='utf-8')
    except (OSError, ValueError) as error:
        parser.error(str(error))
    complete = sum(row['complete_group'] for row in rows)
    print(f'Complete groups: {complete}\nIncomplete groups: {len(rows) - complete}')
    for name in ('index.html', 'viewer_manifest.csv', 'missing_files.csv'):
        print(output / name)


if __name__ == '__main__':
    main()
