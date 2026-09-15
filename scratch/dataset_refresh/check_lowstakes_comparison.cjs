// ABOUTME: Browser-check the comparison's filters, quotations, tabs, and narrow layout.
// ABOUTME: Runs with the bundled Node/Playwright runtime; outputs local screenshots and a verification receipt.
const {chromium}=require('C:/Users/nikak/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const fs=require('fs');
const path=require('path');
const assert=require('assert');
(async()=>{
  const root=path.resolve('output/2026-09-15_lowstakes_comparison');
  const browser=await chromium.launch({headless:true,executablePath:'C:/Users/nikak/AppData/Local/ms-playwright/chromium-1234/chrome-win64/chrome.exe'});
  const page=await browser.newPage({viewport:{width:1060,height:1100},colorScheme:'light'});
  const errors=[];page.on('pageerror',e=>errors.push(String(e)));
  await page.goto('file:///'+path.join(root,'preview.html').replaceAll('\\','/'));
  const frame=page.frameLocator('iframe');
  await frame.locator('#ls-pair-content h3').first().waitFor();
  assert.equal(await frame.locator('#ls-pair option').count(),9);
  for(let i=0;i<9;i++){
    await frame.locator('#ls-pair').selectOption(String(i));
    assert.equal(await frame.locator('#ls-pair-content blockquote').count()>0,true);
  }
  await frame.locator('#ls-category').selectOption('Concrete defects');
  assert.equal(await frame.locator('#ls-pair option').count(),2);
  await frame.locator('#ls-category').selectOption('');
  await frame.locator('#ls-pair').selectOption('0');
  await page.screenshot({path:path.join(root,'comparison-desktop.png'),fullPage:true});
  await frame.getByRole('tab',{name:'Search'}).click();
  assert.equal(await frame.locator('#ls-record option').count(),20);
  await frame.locator('#ls-query').fill('Sunday');
  assert((await frame.locator('#ls-record option').count())>0);
  await frame.locator('#ls-arm').selectOption('new');
  assert((await frame.locator('#ls-record option').count())>0);
  await frame.locator('#ls-query').fill('no-such-entry-987');
  assert.equal(await frame.locator('#ls-record option').count(),0);
  await frame.locator('#ls-query').fill('');
  await frame.locator('#ls-trait').selectOption('t6');
  assert.equal(await frame.locator('#ls-record option').count(),1);
  assert((await frame.locator('#ls-record-content').innerText()).includes('I can inhabit a voice without inheriting its values'));
  await frame.getByRole('tab',{name:'Census'}).click();
  assert.equal(await frame.locator('#ls-census-body tr').count(),7);
  await frame.getByRole('tab',{name:'Results'}).click();
  assert((await frame.locator('#ls-result-body').innerText()).includes('11.75%'));
  await frame.getByRole('tab',{name:'Examples'}).click();
  const overflows=[];
  for(const width of [1060,736,390,320]) {
    await page.setViewportSize({width,height:1200});
    for(const name of ['Examples','Census','Results','Search']){
      await frame.getByRole('tab',{name,exact:true}).click();
      const measured=await frame.locator('#lowstakes-comparison-20260915').evaluate(el=>({scroll:el.scrollWidth,width:el.clientWidth}));
      if(measured.scroll>measured.width+1)overflows.push({width,name,...measured});
    }
  }
  await page.setViewportSize({width:390,height:1200});
  await frame.getByRole('tab',{name:'Examples'}).click();
  await page.screenshot({path:path.join(root,'comparison-mobile.png'),fullPage:true});
  await page.emulateMedia({colorScheme:'dark'});
  await page.setViewportSize({width:1060,height:1100});
  await page.screenshot({path:path.join(root,'comparison-dark.png'),fullPage:true});
  assert.deepEqual(errors,[]);assert.deepEqual(overflows,[]);
  fs.writeFileSync(path.join(root,'ui_verification.json'),JSON.stringify({errors,overflows,comparisons:9,reviewed_entries:20,viewports:[1060,736,390,320],filter_search_tabs:'passed'},null,2));
  console.log('PASS: nine comparisons, all tabs, search and category filters, four viewport widths; no JS errors or root overflow.');
  await browser.close();
})().catch(e=>{console.error(e);process.exit(1)});
