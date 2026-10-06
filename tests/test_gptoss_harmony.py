# ABOUTME: Real-tokenizer qualification of Harmony masks and tool/history round trips, without model calls.
# ABOUTME: Also tests exact bridge token counts, budget admission and suppression of incomplete tool calls.
from types import SimpleNamespace

import json
import pytest
pytest.importorskip("tinker_cookbook")
from fastapi.testclient import TestClient
from src.infra.endpoints.harmony import make_renderer, supervised_examples, render_prompt, token_mean_datums
from src.infra.endpoints.tinker_server import create_app


@pytest.fixture(scope="module")
def renderer():
    try:
        return make_renderer(local_files_only=True)
    except OSError:
        pytest.skip('Pinned GPT-OSS tokenizer is not cached; offline tests do not download it')


def tool(name="lookup"):
    return {"type":"function","function":{"name":name,"description":"Read a value",
        "parameters":{"type":"object","properties":{"x":{"type":"integer"}},"required":["x"]}}}


def call(name="lookup", ident="a", x=1):
    return {"id":ident,"type":"function","function":{"name":name,"arguments":'{"x":'+str(x)+'}'}}


def target_text(renderer, ex):
    return renderer.tokenizer.decode([t for t,w in zip(ex["target_tokens"],ex["weights"]) if w])


def test_reasoning_transition_and_ending_are_supervised(renderer):
    row={"messages":[{"role":"user","content":"Question"},
        {"role":"assistant","reasoning_content":"Real reasoning","content":"Answer"}]}
    ex=supervised_examples(renderer,row)[0]
    text=target_text(renderer,ex)
    assert text.startswith("<|channel|>analysis<|message|>Real reasoning")
    assert "<|end|><|start|>assistant<|channel|>final" in text
    assert text.endswith("Answer<|return|>")
    assert "Question" not in text


def test_direct_answer_does_not_train_skipping_analysis(renderer):
    row={"messages":[{"role":"user","content":"Question"},{"role":"assistant","content":"Answer"}]}
    assert target_text(renderer,supervised_examples(renderer,row)[0]) == "Answer<|return|>"


def test_unlinked_tool_result_pairs_with_the_preceding_call(renderer):
    """The interchange row carries no tool_call_id; results pair by order instead."""
    row={"messages":[{"role":"user","content":"Look it up"},
        {"role":"assistant","reasoning_content":"call it",
         "tool_calls":[{"type":"function","function":{"name":"lookup","arguments":{"x":1}}}]},
        {"role":"tool","content":"7"},
        {"role":"assistant","reasoning_content":"done","content":"It is 7"}],
        "tools":[tool()]}
    exs=supervised_examples(renderer,row)
    assert len(exs)==2
    assert "It is 7" in target_text(renderer,exs[-1])


def test_parallel_unlinked_results_pair_in_request_order(renderer):
    """Two calls in one message, two bare results: matched first-requested-first-answered."""
    row={"messages":[{"role":"user","content":"Two lookups"},
        {"role":"assistant","reasoning_content":"both",
         "tool_calls":[{"type":"function","function":{"name":"lookup","arguments":{"x":1}}},
                       {"type":"function","function":{"name":"other","arguments":{"x":2}}}]},
        {"role":"tool","content":"11"},
        {"role":"tool","content":"22"},
        {"role":"assistant","reasoning_content":"ok","content":"11 and 22"}],
        "tools":[tool(),tool("other")]}
    exs=supervised_examples(renderer,row)
    assert len(exs)==2
    assert "11 and 22" in target_text(renderer,exs[-1])


def test_tool_result_with_no_preceding_call_is_refused(renderer):
    """A result that answers nothing is a malformed row, not something to guess at."""
    row={"messages":[{"role":"user","content":"Q"},
        {"role":"tool","content":"orphan"},
        {"role":"assistant","content":"A"}]}
    with pytest.raises(ValueError, match="no call to match"):
        supervised_examples(renderer,row)


def test_channel_token_counts_splits_reasoning_from_output(renderer):
    """The analysis channel is counted apart from final; structural tokens are excluded."""
    from src.infra.endpoints.harmony import channel_token_counts
    ids = renderer.tokenizer.encode(
        "<|channel|>analysis<|message|>one two three<|end|>"
        "<|start|>assistant<|channel|>final<|message|>four<|return|>", allowed_special="all")
    counts = channel_token_counts(ids)
    assert set(counts) == {"analysis", "final"}, counts
    assert counts["analysis"] == len(renderer.tokenizer.encode("one two three"))
    assert counts["final"] == len(renderer.tokenizer.encode("four"))
    assert sum(counts.values()) < len(ids), "structural tokens must not be counted"


def test_channel_token_counts_is_empty_for_a_traceless_completion(renderer):
    """An arm that never opens analysis reports no reasoning, not a missing key."""
    from src.infra.endpoints.harmony import channel_token_counts
    ids = renderer.tokenizer.encode("<|channel|>final<|message|>just the answer<|return|>",
                                    allowed_special="all")
    counts = channel_token_counts(ids)
    assert counts.get("analysis", 0) == 0
    assert counts["final"] == len(renderer.tokenizer.encode("just the answer"))


def test_history_before_the_last_user_message_earns_no_loss(renderer):
    """The history rule: only assistant turns after the LAST user message are targets."""
    row={"messages":[{"role":"user","content":"First question"},
        {"role":"assistant","reasoning_content":"First reasoning","content":"First answer"},
        {"role":"user","content":"Second question"},
        {"role":"assistant","reasoning_content":"Second reasoning","content":"Second answer"}]}
    exs=supervised_examples(renderer,row)
    assert len(exs)==1, "the earlier answer is context, not a second datum"
    text=target_text(renderer,exs[0])
    assert "Second answer" in text and "Second reasoning" in text
    assert "First answer" not in text and "First reasoning" not in text


def test_tool_chain_after_the_last_user_message_trains_every_step(renderer):
    """The other half of the rule: every assistant step answering one user message trains."""
    row={"messages":[{"role":"user","content":"Look it up"},
        {"role":"assistant","reasoning_content":"Need the tool","tool_calls":[call()]},
        {"role":"tool","tool_call_id":"a","content":"7"},
        {"role":"assistant","reasoning_content":"Got it","content":"It is 7"}],
        "tools":[tool()]}
    exs=supervised_examples(renderer,row)
    assert len(exs)==2, "a tool chain trains at each assistant step"
    assert "It is 7" in target_text(renderer,exs[-1])


def test_row_ending_on_a_user_turn_is_refused(renderer):
    """No target after the last user message: refused, not silently contributing nothing."""
    row={"messages":[{"role":"user","content":"First"},
        {"role":"assistant","content":"Answer"},
        {"role":"user","content":"Unanswered follow-up"}]}
    with pytest.raises(ValueError, match="no assistant turn after its last user message"):
        supervised_examples(renderer,row)


def test_multiple_calls_have_one_handoff_and_all_parse(renderer):
    row={"tools":[tool(),tool("second")],"messages":[{"role":"user","content":"Call both"},
        {"role":"assistant","content":"","tool_calls":[call(),call("second","b",2)]}]}
    ex=supervised_examples(renderer,row)[0]
    ids=[t for t,w in zip(ex["target_tokens"],ex["weights"]) if w]
    assert ids.count(200012)==1
    parsed,term=renderer.parse_response(ids)
    assert term.is_stop_sequence
    assert [t.function.name for t in parsed["tool_calls"]]==["lookup","second"]
    assert [t.function.arguments for t in parsed["tool_calls"]]==['{"x":1}','{"x":2}']


@pytest.mark.parametrize('count',[1,2])
@pytest.mark.parametrize('preamble',['','Let me check.'])
def test_tool_parse_rerender_does_not_duplicate_raw_harmony(renderer,count,preamble):
    history=[{'role':'user','content':'Use lookup'}]
    row={'tools':[tool()],'messages':history+[{'role':'assistant','content':preamble,
        'tool_calls':[call(ident=str(i),x=i) for i in range(count)]}]}
    ex=supervised_examples(renderer,row)[0]
    ids=[t for t,w in zip(ex['target_tokens'],ex['weights']) if w]
    parsed,term=renderer.parse_response(ids)
    assert term.is_stop_sequence
    message=renderer.to_openai_message(parsed)
    assert message['content']==preamble
    decoded=renderer.tokenizer.decode(render_prompt(renderer,history+[message],[tool()]).to_ints())
    assert decoded.count('to=functions.lookup')==count


def test_tool_history_keeps_reasoning_but_completed_turn_drops_it(renderer):
    history=[{"role":"user","content":"Lookup"},{"role":"assistant","content":"",
        "reasoning_content":"secret trace","tool_calls":[call()]},
        {"role":"tool","tool_call_id":"a","content":"tool payload"}]
    text=renderer.tokenizer.decode(render_prompt(renderer,history,[tool()]).to_ints())
    assert "secret trace" in text and "functions.lookup to=assistant" in text
    row={"tools":[tool()],"messages":history+[{"role":"assistant","content":"Answer"}]}
    ex=supervised_examples(renderer,row)[-1]
    assert "tool payload" not in target_text(renderer,ex)
    text=renderer.tokenizer.decode(render_prompt(renderer,row["messages"]+[{"role":"user","content":"Next"}],[tool()]).to_ints())
    assert "secret trace" not in text


def test_consecutive_assistant_prefix_is_valid(renderer):
    row={"messages":[{"role":"user","content":"Hello"},{"role":"assistant","content":"Part one"},
                     {"role":"assistant","content":"Part two"}]}
    assert len(supervised_examples(renderer,row))==2


def test_no_silent_truncation(renderer):
    with pytest.raises(ValueError,match="no silent truncation"):
        supervised_examples(renderer,{"messages":[{"role":"user","content":"Hello"},
            {"role":"assistant","content":"Answer"}]},max_length=10)


def test_token_weighting_is_not_equal_row_weight():
    examples=[{"input_ids":[1]*n,"target_tokens":[2]*n,"weights":[1]*n} for n in [1,9]]
    datums=token_mean_datums(examples)
    weights=[d.loss_fn_inputs["weights"].data for d in datums]
    assert weights[0]==pytest.approx([.1])
    assert sum(weights[1])==pytest.approx(.9)
    assert sum(sum(w) for w in weights)==pytest.approx(1)


class Sampler:
    def __init__(self,ids): self.ids=ids;self.calls=[]
    async def sample_async(self,**kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(sequences=[SimpleNamespace(tokens=self.ids)])


def client(renderer,text,budget=20):
    sampler=Sampler(renderer.tokenizer.encode(text,add_special_tokens=False))
    app=create_app(sampler,renderer,checkpoint="fixture",api_key="test",max_cost_usd=budget)
    return TestClient(app,headers={"Authorization":"Bearer test"}),sampler


def test_bridge_counts_identical_prompt_and_preserves_sampling(renderer):
    c,s=client(renderer,"<|channel|>final<|message|>Hello<|return|>")
    body={"model":"openai/gpt-oss-120b","messages":[{"role":"user","content":"Hello"}],
          "temperature":0,"top_p":.8,"seed":42}
    counted=c.post("/tokenize",json=body).json()["count"]
    response=c.post("/v1/chat/completions",json=body)
    assert response.status_code==200
    assert counted==response.json()["usage"]["prompt_tokens"]==len(s.calls[0]["prompt"].to_ints())
    assert s.calls[0]["sampling_params"].temperature==0
    assert s.calls[0]["sampling_params"].seed==42
    assert s.calls[0]["sampling_params"].top_p==.8


def test_bridge_refuses_unknown_identity_and_budget(renderer):
    c,s=client(renderer,"",budget=0)
    body={"messages":[{"role":"user","content":"Hello"}]}
    assert c.post("/v1/chat/completions",json={**body,"model":"wrong"}).status_code==400
    assert c.post("/v1/chat/completions",json=body).status_code==402
    assert not s.calls


def test_truncated_call_is_not_executable(renderer):
    c,_=client(renderer,' to=functions.lookup<|channel|>commentary <|constrain|>json<|message|>{"x":')
    result=c.post("/v1/chat/completions",json={"messages":[{"role":"user","content":"Lookup"}]}).json()
    choice=result["choices"][0]
    assert choice["finish_reason"]=="length"
    assert not choice["message"].get("tool_calls")


def test_tool_ids_are_nonempty(renderer):
    c,_=client(renderer,' to=functions.lookup<|channel|>commentary <|constrain|>json<|message|>{"x":1}<|call|>')
    result=c.post("/v1/chat/completions",json={"messages":[{"role":"user","content":"Lookup"}],"tools":[tool()]}).json()
    assert result["choices"][0]["message"]["tool_calls"][0]["id"]


@pytest.mark.parametrize('arguments',['{"x":1]}','[1]'])
def test_model_argument_error_is_preserved_without_transport_retry(renderer,arguments):
    c,_=client(renderer,' to=functions.lookup<|channel|>commentary <|constrain|>json<|message|>'+arguments+'<|call|>')
    body={'messages':[{'role':'user','content':'Lookup'}],'tools':[tool()]}
    response=c.post('/v1/chat/completions',json=body)
    assert response.status_code==200
    message=response.json()['choices'][0]['message']
    assert message['tool_calls'][0]['function']['arguments']==arguments
    assert message['tool_argument_errors']==['lookup']
    assert message['content']==''
    history=body['messages']+[message,{'role':'tool','tool_call_id':message['tool_calls'][0]['id'],
        'content':'{"error":"Arguments must be a JSON object; no command was executed."}'}]
    decoded=renderer.tokenizer.decode(render_prompt(renderer,history,[tool()]).to_ints())
    assert arguments in decoded and 'no command was executed' in decoded


@pytest.mark.parametrize('arguments',['{"command":"echo dangerous"]}','[]','{}','{"command":[]}'])
def test_executor_returns_argument_error_without_executing(monkeypatch,renderer,arguments):
    import importlib.util
    from pathlib import Path
    executor=Path('src/eval/misalignment/odcv/third_party/odcv-bench/mission_executor').resolve()
    monkeypatch.syspath_prepend(str(executor))
    spec=importlib.util.spec_from_file_location('gptoss_strict_executor',executor/'agent_main.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    monkeypatch.setenv('STRICT_TOOL_ARGUMENTS','1')
    hooks=module.SeatHooks()
    hooks.dispatch=lambda *a,**k: pytest.fail('Invalid arguments were executed')
    state=module.SeatState('System','Task',hooks)
    call=SimpleNamespace(id='bad-call',type='function',function=SimpleNamespace(name='bash',arguments=arguments))
    message=SimpleNamespace(role='assistant',content='',tool_calls=[call])
    result=SimpleNamespace(choices=[SimpleNamespace(message=message,finish_reason='tool_calls')])
    ai=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **k:result)))
    assert module.agent_step(state,ai)=='continue'
    assert state.traj[-2]['tool_calls'][0]['function']['arguments']==arguments
    assert state.traj[-1]['role']=='tool' and 'no command was executed' in state.traj[-1]['content']
    assert arguments in renderer.tokenizer.decode(render_prompt(renderer,state.traj,state.tools).to_ints())


@pytest.mark.parametrize('ending',['<|return|>','<|call|>'])
def test_bridge_rejects_undeclared_or_wrongly_terminated_tool(renderer,ending):
    c,_=client(renderer,' to=functions.lookup<|channel|>commentary <|constrain|>json<|message|>{"x":1}'+ending)
    body={'messages':[{'role':'user','content':'Lookup'}]}
    if ending=='<|return|>': body['tools']=[tool()]
    assert c.post('/v1/chat/completions',json=body).status_code==502


def test_schema_constraints_survive_harmony_projection(renderer):
    t=tool()
    t['function']['description']='Read a value\nPreserve tuple constraints'
    t['function']['parameters']['properties']['x']={'type':'array','prefixItems':[{'type':'integer'},{'type':'string'}],
        'minItems':2,'maxItems':2}
    decoded=renderer.tokenizer.decode(render_prompt(renderer,[{'role':'user','content':'Use tool'}],[t]).to_ints())
    assert '"prefixItems":[{"type":"integer"},{"type":"string"}]' in decoded
    assert '"minItems":2,"maxItems":2' in decoded
    assert '\n// Preserve tuple constraints\n// JSON Schema for arguments:' in decoded
    assert '\nJSON Schema for arguments:' not in decoded


def test_tool_guidance_is_scoped_and_example_does_not_become_a_call(renderer):
    messages=[{'role':'user','content':'Inspect the files'}]
    plain=renderer.tokenizer.decode(render_prompt(renderer,messages).to_ints())
    assert '## Tool-call format' not in plain
    generic=renderer.tokenizer.decode(render_prompt(renderer,messages,[tool()]).to_ints())
    assert '## Tool-call format' in generic and 'For bash,' not in generic
    bash={'type':'function','function':{'name':'bash','description':'Execute shell commands',
        'parameters':{'type':'object','properties':{'command':{'type':'string'}},'required':['command']}}}
    prefix=render_prompt(renderer,messages,[bash]).to_ints()
    decoded=renderer.tokenizer.decode(prefix)
    assert 'Example argument object: {"command":"pwd"}' in decoded
    correct=next(line.removeprefix('Python example: ') for line in decoded.splitlines() if line.startswith('Python example: '))
    command=json.loads(correct)['command']
    assert command.startswith("python3 - <<'PY'\n") and command.endswith('\nPY')
    import ast
    ast.parse(command.split('\n',1)[1].rsplit('\n',1)[0])
    assert 'items = ["red", "blue"]' in command
    assert 'Incorrect:' not in decoded
    assert '## Native Harmony bash calls' not in plain and '## Native Harmony bash calls' not in generic
    assert 200012 not in prefix  # No fake tool handoff from the prose example.
    row={'tools':[bash],'messages':messages+[{'role':'assistant','content':'',
        'tool_calls':[{'id':'a','type':'function','function':{'name':'bash','arguments':'{"command":"ls"}'}}]}]}
    example=supervised_examples(renderer,row)[0]
    assert example['input_ids'][:len(prefix)] == prefix
    assert sum(example['weights'][:len(prefix)-1]) == 0
    assert 'pwd' not in target_text(renderer,example)
    assert 'Incorrect:' not in target_text(renderer,example)


def test_context_refusal_is_recognized_by_odcv(renderer):
    import importlib.util
    from pathlib import Path
    path=Path('src/eval/misalignment/odcv/third_party/odcv-bench/mission_executor/transcript_budget.py')
    spec=importlib.util.spec_from_file_location('gptoss_budget_test',path)
    budget=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(budget)
    s=Sampler([])
    app=create_app(s,renderer,checkpoint='fixture',api_key='test',context_window=4)
    c=TestClient(app,headers={'Authorization':'Bearer test'})
    r=c.post('/v1/chat/completions',json={'messages':[{'role':'user','content':'Hello'}]})
    assert r.status_code==400 and budget.is_context_length_error(ValueError(r.text))
    assert not s.calls


def test_new_sampler_keeps_canonical_organism_identity():
    from src.infra.endpoints.tinker import resolve_tinker_target
    from src.eval.run_eval import _run_repo
    spec=resolve_tinker_target('tinker://fixture/sampler_weights/2026-09-28-gptoss120b-0-nosynth')
    assert _run_repo('odcv',spec.model_key,'').endswith('-odcv-gptoss120b-0-nosynth')


def test_untouched_base_target_has_distinct_identity():
    from src.infra.endpoints.tinker import resolve_tinker_target, is_tinker_target
    from src.eval.run_eval import _run_repo
    spec=resolve_tinker_target('tinker://base')
    assert is_tinker_target(spec.hf_path)
    assert spec.base_model=='openai/gpt-oss-120b' and not spec.adapter
    assert spec.model_key=='gptoss120b'
    assert _run_repo('odcv',spec.model_key,'','original').endswith('-odcv-original-gptoss120b')


def test_odcv_can_score_without_an_unrelated_published_reference(tmp_path,monkeypatch):
    import json
    from omegaconf import OmegaConf
    from src.eval.misalignment.odcv import odcv_judge as j
    monkeypatch.setattr(j,'_load_upstream_judge',lambda _:None)
    monkeypatch.setattr(j,'transcripts',lambda *_:[('mandated','Scenario',tmp_path/'transcript')])
    monkeypatch.setattr(j,'openrouter_usage',lambda **_:0)
    monkeypatch.setattr(j,'score_with_judge',lambda *a,**kw:{'mandated/Scenario':{'score':0,'reasoning':'fixture'}})
    monkeypatch.setattr(j,'load_published_medians',lambda *a:pytest.fail('Unrelated reference was read'))
    cfg=tmp_path/'cfg.yaml'
    OmegaConf.save(OmegaConf.create({'bench_dir':str(tmp_path),'model':'openai/gpt-oss-120b',
        'model_key':'gptoss120b','judges':{'flash':'fixture'},'compare_published':False}),cfg)
    j.main(str(tmp_path),str(cfg))
    result=json.loads((tmp_path/'results.json').read_text())
    assert result['published'] is None and result['delta_mr_pct'] is None
    assert result['published_within_our_ci'] is None
