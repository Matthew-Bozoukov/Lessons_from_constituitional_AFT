// ABOUTME: Offline contract, missing-data, protocol and signed-contrast regression tests.
// ABOUTME: Synthetic fixtures exercise generic comparisons without baking research into the UI.
import assert from "node:assert/strict";
import test from "node:test";
import { parseComparison, compatible, difference, listComparisons, loadComparison, traitDiffers } from "../lib/comparisons.ts";

const pin="a".repeat(40);
function fixture() {
  return {schema_version:1,title:"Synthetic test fixture",summary:"Not research data",limitations:["Fixture only"],
    traits:[{id:"length",label:"Words",description:"Mean words"},{id:"stakes",label:"Stakes",description:"Measured magnitude"}],
    metrics:[{id:"error",label:"Errors",unit:"%",lower_is_better:true}],
    arms:[10,30].map((v,i)=>({id:`arm${i}`,label:`Arm ${i}`,dataset:{repo:`test/data${i}`,revision:pin,file:"data.jsonl",subset:"Full dataset",row_count:10},
      model:{repo:`test/model${i}`,revision:pin,base_revision:pin,seed:0},
      evaluation:{repo:`test/eval${i}`,revision:pin,protocol:{benchmark:"test",scenario_set:"same",temperature:0.7},repeats:3,metrics:{error:{value:v,numerator:v/10,denominator:10}}},
      traits:{length:{value:100+i,basis:"measured",evidence:[{label:"Audit",url:"https://example.org/audit"}]},stakes:{value:null,basis:"unmeasured",evidence:[]}}})),
    contrasts:[{baseline:"arm0",arm:"arm1",metric:"error",delta:20,interval:{low:5,high:35,method:"paired scenarios"}}]};
}
test("reference reversal reverses both point estimate and paired interval",()=>{
  const d=parseComparison(fixture());
  assert.equal(difference(d,d.arms[0],d.arms[1],"error").value,20);
  assert.deepEqual(difference(d,d.arms[1],d.arms[0],"error"),{value:-20,interval:{low:-35,high:-5,method:"paired scenarios"}});
});
test("protocol key order is irrelevant; changed settings forbid differences",()=>{
  const d=fixture();d.arms[1].evaluation.protocol={temperature:0.7,scenario_set:"same",benchmark:"test"};
  assert.ok(compatible(...d.arms));
  d.arms[1].evaluation.protocol.temperature=0.8;
  assert.equal(difference(d,...d.arms,"error"),null);
  assert.throws(()=>parseComparison(d),/compatible/);
  d.contrasts=[];assert.doesNotThrow(()=>parseComparison(d));
});
test("unknown is not zero; missing outcome never becomes zero or a difference",()=>{
  const d=fixture();d.contrasts=[];delete d.arms[1].evaluation.metrics.error;
  assert.doesNotThrow(()=>parseComparison(d));
  assert.equal(difference(d,...d.arms,"error"),null);
  assert.equal(traitDiffers(d,"stakes"),false);
  assert.equal(traitDiffers(d,"length"),true);
  d.arms[1].traits.stakes.value=0;
  assert.throws(()=>parseComparison(d),/unknown traits/);
});
test("invalid or misleading research records fail closed",()=>{
  for (const mutate of [d=>d.schema_version=2,d=>d.arms[0].dataset.revision="main",d=>d.arms[0].evaluation.metrics.error.value=11,
    d=>d.arms[0].traits.length.evidence=[],d=>d.contrasts[0].delta=21,d=>d.contrasts[0].interval.low=25,
    d=>d.arms[0].traits.length.evidence[0].url="javascript:alert(1)",d=>d.arms[1].id="arm0"]) {
    const d=fixture();mutate(d);assert.throws(()=>parseComparison(d),/Invalid comparison/);
  }
});
test("discovery follows pagination and loads the exact listed revision",async t=>{
  const urls=[];
  t.mock.method(globalThis,"fetch",async url=>{
    urls.push(String(url));
    if (String(url).includes("/resolve/")) return Response.json(fixture());
    if (String(url).includes("page=2")) return Response.json([{id:"test/second",sha:pin}]);
    return Response.json([{id:"test/first",sha:pin,cardData:{pretty_name:"First comparison"}}],{headers:{Link:'<https://huggingface.co/api/datasets?page=2>; rel="next"'}});
  });
  const rows=await listComparisons("fixture-pagination");assert.equal(rows.length,2);
  assert.equal(rows[0].title,"First comparison");await loadComparison(rows[0]);
  assert.ok(urls[0].includes("filter=dataset-model-comparison"));
  assert.ok(urls.at(-1).includes(`/resolve/${pin}/results/dataset_comparison.json`));
});
test("network errors are errors, not an empty research collection",async t=>{
  t.mock.method(globalThis,"fetch",async()=>new Response("Unavailable",{status:503}));
  await assert.rejects(listComparisons("fixture-network-error"),/503/);
});

test("the populated view renders counts, evidence, uncertainty and missing outcomes",async()=>{
  const {registerHooks}=await import("node:module");
  const {readFileSync}=await import("node:fs");
  const {fileURLToPath}=await import("node:url");
  const ts=await import("typescript");
  const hooks=registerHooks({
    resolve(specifier,context,next){
      if (specifier==="@/lib/comparisons") return {url:new URL("../lib/comparisons.ts",import.meta.url).href,shortCircuit:true};
      return next(specifier,context);
    },
    load(url,context,next){
      if (!url.endsWith(".tsx")) return next(url,context);
      return {format:"module",shortCircuit:true,source:ts.transpileModule(readFileSync(fileURLToPath(url),"utf8"),
        {compilerOptions:{jsx:ts.JsxEmit.ReactJSX,module:ts.ModuleKind.ESNext}}).outputText};
    },
  });
  try {
    const {ComparisonView}=await import("../app/components/ComparisonExplorer.tsx");
    const {createElement}=await import("react"), {renderToStaticMarkup}=await import("react-dom/server");
    const d=fixture();
    let html=renderToStaticMarkup(createElement(ComparisonView,{data:parseComparison(d)}));
    assert.match(html,/1\/10/);assert.match(html,/paired 95% CI/);assert.match(html,/Measured/);
    assert.match(html,/https:\/\/example.org\/audit/);assert.match(html, /Reference/);
    d.contrasts=[];delete d.arms[1].evaluation.metrics.error;
    html=renderToStaticMarkup(createElement(ComparisonView,{data:parseComparison(d)}));
    assert.match(html,/Not reported/);assert.match(html,/Outcome missing/);
    assert.doesNotMatch(html,/NaN|undefined/);
  } finally {hooks.deregister();}
});
