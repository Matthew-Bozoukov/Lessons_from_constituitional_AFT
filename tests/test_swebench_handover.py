# ABOUTME: CPU-only handover tests with real orphaned processes and mocked paid-provider calls.
# ABOUTME: Check PID identity, bounded reaper grace, unchanged ledgers and adopted-worker cleanup.
import os
import unittest
if os.name != 'posix':
    raise unittest.SkipTest('The coordinator runs on Linux')

import json
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
from unittest.mock import Mock, patch
from concurrent.futures import Future

from omegaconf import OmegaConf
import psutil
from src.eval.capabilities.swebench_mini import fleet, fleet_handover as handover
from src.eval.capabilities.swebench_mini.fleet_state import atomic, read


class HandoverTests(unittest.TestCase):
    def test_adoption_does_not_extend_an_unready_workers_startup_allowance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);cfg=OmegaConf.create({'root':tmp,'port_base':8100,'ssh_key':'test','cleanup_reserve_seconds':180,'boot_seconds':900})
            record={'slot':2,'id':'gpu','worker_pid':123,'worker_created':1,'created':time.time()-901,'expires':time.time()+400,'server':'root@host:22'}
            atomic(root/'metadata/state.json',{'pods':[record|{'status':'serving'}],'tasks':{},'halt':None})
            worker=Mock();worker.poll.return_value=None
            with patch.object(handover,'validate_worker',return_value=worker),patch.object(fleet.runpod,'start_watchdog'),patch.object(fleet.runpod,'teardown') as teardown,patch.object(fleet,'SshExec') as ssh,patch.object(fleet,'stop_process'),patch.object(fleet.session,'recover_worker'):
                ssh.return_value._ssh.return_value='server log'
                with self.assertRaisesRegex(TimeoutError,'startup allowance'):
                    handover.monitor(cfg,root/'launch.yaml',record,{'campaign':'ours'})
            teardown.assert_called_once_with('gpu')

    def test_budget_deferral_does_not_trigger_gpu_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);cfg=OmegaConf.load('configs/eval/swebench_mini/lite.yaml');cfg.root=tmp
            cfg.allocation_retry_seconds=cfg.allocation_retry_max_seconds=0
            cfg.allocation_fallback_after_attempts=1;cfg.allocation_fallback_after_seconds=0
            atomic(root/'metadata/state.json',{'pods':[],'tasks':{'a':{'status':'pending','attempts':[]}},'halt':None,'deadline':None})
            choices=[]
            def submit(fn,config,*args):
                choices.append(config.gpu);result=Future()
                if len(choices)==1:
                    result.set_exception(RuntimeError('Allocation deferred: cumulative budget reservation unavailable'))
                else:
                    saved=read(root/'metadata/state.json');saved['tasks']['a']['status']='valid';atomic(root/'metadata/state.json',saved);result.set_result(None)
                return result
            pool=Mock();pool.submit.side_effect=submit
            with patch.object(fleet,'ThreadPoolExecutor') as executor,patch.object(fleet,'price_ceiling',return_value=3.35),patch.object(fleet.time,'sleep'),patch.object(fleet.session,'checkpoints'),patch.object(fleet,'reconcile_rejections'),patch.object(fleet.shutil,'disk_usage',return_value=Mock(free=200*2**30)),patch.object(fleet.psutil,'virtual_memory',return_value=Mock(available=180*2**30)):
                executor.return_value.__enter__.return_value=pool
                fleet.phase(cfg,root/'config.yaml',['a'],1,cfg.rental_seconds,{'budget_usd':280})
            self.assertEqual(choices,[cfg.gpu,cfg.gpu])

    def test_adopted_replica_occupies_a_fleet_lane_before_new_rental(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);cfg=OmegaConf.load('configs/eval/swebench_mini/lite.yaml');cfg.root=tmp
            record={'slot':2,'id':'gpu','created':time.time(),'expires':time.time()+10000,'ceiling_hourly':3.35}
            atomic(root/'metadata/state.json',{'pods':[record],'tasks':{'a':{'status':'running','attempts':[]},'b':{'status':'pending','attempts':[]}},'halt':None,'deadline':None})
            adopted_future=Future();calls=[]
            def submit(fn,*args):
                calls.append(fn)
                if fn is handover.monitor:return adopted_future
                saved=read(root/'metadata/state.json')
                for task in saved['tasks'].values():task['status']='valid'
                atomic(root/'metadata/state.json',saved)
                adopted_future.set_result(None)
                result=Future();result.set_result(None);return result
            pool=Mock();pool.submit.side_effect=submit
            with patch.object(fleet,'ThreadPoolExecutor') as executor,patch.object(fleet,'price_ceiling',return_value=3.35),patch.object(fleet.time,'sleep'),patch.object(fleet.session,'checkpoints'),patch.object(fleet,'reconcile_rejections'),patch.object(fleet.shutil,'disk_usage',return_value=Mock(free=200*2**30)),patch.object(fleet.psutil,'virtual_memory',return_value=Mock(available=180*2**30)):
                executor.return_value.__enter__.return_value=pool
                fleet.phase(cfg,root/'config.yaml',['a','b'],2,cfg.rental_seconds,{'budget_usd':280},adopted=[record])
            self.assertEqual(calls,[handover.monitor,fleet.replica])

    def test_real_worker_survives_parent_exit_and_can_be_fenced(self):
        with tempfile.TemporaryDirectory() as tmp:
            file = Path(tmp)/'pid.json'
            code = ('import subprocess,sys,time,json\nfrom pathlib import Path\n'
                    'p=subprocess.Popen([sys.executable,"-c","import time; time.sleep(60)"],start_new_session=True)\n'
                    f'Path({str(file)!r}).write_text(json.dumps(p.pid))\n'
                    'time.sleep(60)\n')
            parent = subprocess.Popen([sys.executable,'-c',code], start_new_session=True)
            worker = None
            try:
                until = time.time()+5
                while not file.exists() and time.time()<until:time.sleep(.02)
                pid = json.loads(file.read_text())
                process = psutil.Process(pid)
                worker = handover.ExistingWorker(pid,process.create_time())
                parent.kill();parent.wait(timeout=5)
                self.assertIsNone(worker.poll())
                with self.assertRaisesRegex(AssertionError,'reused'):
                    handover.ExistingWorker(pid,process.create_time()-1)
                fleet.stop_process(worker)
                self.assertEqual(worker.poll(),0)
            finally:
                if parent.poll() is None:parent.kill();parent.wait(timeout=5)
                if worker and worker.poll() is None:os.killpg(worker.pid,signal.SIGKILL)

    def test_grace_is_campaign_scoped_and_expires_within_three_minutes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'metadata/coordinator-handover.json'
            valid={'campaign':'ours','status':'prepared','created':100,'expires':220}
            atomic(path,valid)
            self.assertEqual(handover.pending(tmp,{'campaign':'ours'},101),valid)
            for data,now,campaign in [(valid,220,'ours'),(valid,101,'other'),
                                     (valid|{'expires':400},101,'ours'),
                                     (valid|{'status':'adopted'},101,'ours')]:
                atomic(path,data)
                self.assertIsNone(handover.pending(tmp,{'campaign':campaign},now))

    def test_claim_checks_owned_inventory_and_keeps_ledger_and_attempts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);config=root/'launch.yaml'
            cfg=OmegaConf.create({'root':tmp,'replicas':10})
            record={'slot':2,'id':'gpu','worker_pid':123,'worker_created':1,'expires':9999,'server':'root@host:22'}
            state={'pods':[record|{'created':12,'actual_hourly':3.19}],
                   'tasks':{'a':{'status':'running','attempts':[{'id':'kept','worker':'2-0'}]}},'halt':None}
            atomic(root/'metadata/state.json',state)
            data={'campaign':'ours','status':'prepared','created':time.time(),'expires':time.time()+120,
                  'previous_pid':99999999,'previous_created':1,'budget_usd':280,'workers':[record]}
            atomic(root/'metadata/coordinator-handover.json',data)
            manifest={'campaign':'ours','budget_usd':280}
            pod={'id':'gpu','env':{'LASR_POD_OWNER':fleet.runpod.POD_OWNER,'LASR_CAMPAIGN':'ours'}}
            with patch.object(handover,'validate_worker',return_value=Mock()),patch.object(psutil,'pid_exists',return_value=False):
                with self.assertRaisesRegex(AssertionError,'inventory changed'):
                    handover.claim(cfg,config,manifest,[])
                adopted=handover.claim(cfg,config,manifest,[pod])
            self.assertEqual(adopted,[record])
            self.assertEqual(read(root/'metadata/state.json'),state)
            self.assertEqual(read(root/'metadata/coordinator-handover.json')['status'],'adopted')

    def test_reaper_grace_does_not_extend_provider_deadline(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);cfg=OmegaConf.create({'root':tmp})
            atomic(root/'metadata/manifest.json',{'campaign':'ours'})
            atomic(root/'metadata/state.json',{'pods':[],'deadline':None})
            atomic(root/'metadata/coordinator-handover.json',{'status':'prepared','campaign':'ours','created':time.time(),'expires':time.time()+120})
            pods=[{'id':'safe','env':{'LASR_POD_OWNER':fleet.runpod.POD_OWNER,'LASR_CAMPAIGN':'ours','LASR_POD_DEADLINE':str(time.time()+300)}},
                  {'id':'expired','env':{'LASR_POD_OWNER':fleet.runpod.POD_OWNER,'LASR_CAMPAIGN':'ours','LASR_POD_DEADLINE':str(time.time()-1)}}]
            with patch.object(fleet.subprocess,'run',return_value=Mock(returncode=1)),patch.object(fleet.runpod,'active_pods',return_value=pods),patch.object(fleet.runpod,'teardown') as teardown:
                fleet.guard(cfg)
            teardown.assert_called_once_with('expired')

    def test_adopted_worker_keeps_original_expiry_and_preserves_completed_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);cfg=OmegaConf.create({'root':tmp,'port_base':8100,'ssh_key':'test','cleanup_reserve_seconds':180})
            record={'slot':2,'id':'gpu','worker_pid':123,'worker_created':1,'expires':time.time()+400,'server':'root@host:22'}
            state={'pods':[record|{'status':'working'}],'tasks':{'a':{'status':'valid','attempts':[{'worker':'2-0','prediction':'kept'}]}},'halt':None}
            atomic(root/'metadata/state.json',state)
            worker=Mock();worker.poll.return_value=0
            guard=Mock()
            with patch.object(handover,'validate_worker',return_value=worker),patch.object(fleet.runpod,'start_watchdog',return_value=guard) as start,patch.object(fleet.runpod,'teardown') as teardown,patch.object(fleet,'SshExec') as ssh,patch.object(fleet,'stop_process') as stop,patch.object(fleet.session,'recover_worker') as recover:
                ssh.return_value._ssh.return_value='server log'
                handover.monitor(cfg,root/'launch.yaml',record,{'campaign':'ours'})
            self.assertLessEqual(start.call_args.args[1],400)
            stop.assert_called_once_with(worker);recover.assert_called_once_with(cfg,2)
            teardown.assert_called_once_with('gpu');guard.terminate.assert_called_once()
            saved=read(root/'metadata/state.json')
            self.assertEqual(saved['tasks'],state['tasks'])
            self.assertEqual(saved['pods'][0]['expires'],record['expires'])
            self.assertEqual(saved['pods'][0]['status'],'terminated')


if __name__ == '__main__':
    unittest.main()
