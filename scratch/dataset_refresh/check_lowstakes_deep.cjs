// ABOUTME: Exercise the audit's evidence selectors, full-text search, and mobile/theme layout.
// ABOUTME: Checks the inline sample and full 1432-row standalone explorer without network calls.
const {chromium}=require('C:/Users/nikak/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const fs=require('fs'),path=require('path'),assert=require('assert');
(async()=>{
  const out=path.resolve('output/2026-09-17_lowstakes_deep_audit');
  const browser=await chromium.launch({headless:true,executablePath:'C:/Users/nikak/AppData/Local/ms-playwright/chromium-1234/chrome-win64/chrome.exe'});
  const page=await browser.newPage({viewport:{width:1060,height:1100}});
  const errors=[],overflows=[];page.on('pageerror',e=>errors.push(String(e)));
  await page.goto('file:///'+path.join(out,'preview.html').replaceAll('\\','/'));
  const f=page.frameLocator('iframe');
  await f.locator('#ld-comparison h3').waitFor();
  for(let i=0;i<9;i++){await f.locator('#ld-pair').selectOption(String(i));assert((await f.locator('#ld-comparison blockquote').count())>=2);}
  await f.locator('#ld-issue').selectOption('Concrete defects');assert.equal(await f.locator('#ld-pair option').count(),2);
  await f.locator('#ld-issue').selectOption('');
  await f.getByRole('tab',{name:'Stakes',exact:true}).click();
  assert.equal(await f.locator('#ld-stake-row option').count(),3);
  assert((await f.locator('#ld-stake-detail').innerText()).includes('pergola'));
  await f.locator('#ld-stake-filter').selectOption('');assert.equal(await f.locator('#ld-stake-row option').count(),95);
  await f.locator('#ld-stake-filter').selectOption('elevated');assert.equal(await f.locator('#ld-stake-row option').count(),28);
  await f.getByRole('tab',{name:'Browse',exact:true}).click();
  assert.equal(await f.locator('#ld-record option').count(),111);
  await f.locator('#ld-query').fill('Sunday');assert((await f.locator('#ld-record option').count())>0);
  await f.locator('#ld-query').fill('no-such-record-987654');assert.equal(await f.locator('#ld-record option').count(),0);
  assert((await f.locator('#ld-record-detail').innerText()).includes('No matching'));
  await f.locator('#ld-reset').click();await f.locator('#ld-arm').selectOption('new');await f.locator('#ld-trait').selectOption('t6');
  assert.equal(await f.locator('#ld-record option').count(),1);
  for(const width of [1060,736,390,320]){
    await page.setViewportSize({width,height:1100});
    for(const name of ['Compare','Stakes','Corpus','Results','Browse']){
      await f.getByRole('tab',{name,exact:true}).click();
      const size=await f.locator('#lowstakes-deep-20260917').evaluate(el=>({scroll:el.scrollWidth,width:el.clientWidth}));
      if(size.scroll>size.width+1)overflows.push({width,name,...size});
    }
  }
  await page.setViewportSize({width:1060,height:1100});await f.getByRole('tab',{name:'Compare',exact:true}).click();
  await page.screenshot({path:path.join(out,'2026-09-17_audit_desktop.png'),fullPage:true});
  await page.emulateMedia({colorScheme:'dark'});await f.getByRole('tab',{name:'Stakes',exact:true}).click();
  await f.locator('#ld-stake-filter').selectOption('safety');
  await page.screenshot({path:path.join(out,'2026-09-17_audit_dark.png'),fullPage:true});
  await page.setViewportSize({width:390,height:1100});
  await page.screenshot({path:path.join(out,'2026-09-17_audit_mobile.png'),fullPage:true});
  await page.goto('file:///'+path.join(out,'2026-09-17_lowstakes_full_explorer.html').replaceAll('\\','/'));
  await f.locator('#ld-comparison h3').waitFor();await f.getByRole('tab',{name:'Browse',exact:true}).click();
  assert.equal(await f.locator('#ld-record option').count(),1432);
  await f.locator('#ld-feature').selectOption('should_i');assert.equal(await f.locator('#ld-record option').count(),642);
  await f.locator('#ld-arm').selectOption('new');assert.equal(await f.locator('#ld-record option').count(),592);
  await f.locator('#ld-feature').selectOption('answer_label');assert.equal(await f.locator('#ld-record option').count(),100);
  await f.locator('#ld-arm').selectOption('old');assert.equal(await f.locator('#ld-record option').count(),0);
  assert.deepEqual(errors,[]);assert.deepEqual(overflows,[]);
  fs.writeFileSync(path.join(out,'ui_verification.json'),JSON.stringify({errors,overflows,inline_records:111,full_records:1432,reviewed_prompts:95,pairs:9,viewports:[1060,736,390,320],filters_search_tabs:'passed'},null,2));
  await browser.close();console.log('PASS: 9 pairs, 95 reviewed prompt selectors, 111 embedded conversations, 1432-row full browser, exact feature counts; no JS errors or overflow.');
})().catch(e=>{console.error(e);process.exit(1)});
