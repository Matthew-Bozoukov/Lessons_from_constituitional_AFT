# ABOUTME: Resume an owned Tinker archive download with bounded HTTP timeouts and visible progress.
# ABOUTME: Verifies the existing prefix and HTTP range before appending; uses the SDK's safe tar extractor.
import argparse
import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess

from dotenv import load_dotenv
from omegaconf import OmegaConf
import requests
import tinker
from tinker_cookbook.weights import download


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',default='scratch/gptoss_control/control_refresh.yaml')
    p.add_argument('--partial',type=Path)
    args=p.parse_args()
    root=Path(__file__).resolve().parents[2]
    os.chdir(root)
    common=Path(subprocess.check_output(['git','rev-parse','--git-common-dir'],text=True).strip()).resolve()
    load_dotenv(common.parent/'.env')
    cfg=OmegaConf.load(args.config)
    out=root/cfg.output
    archive=out/'checkpoint.tar.part'
    ckpt=json.loads((out/'trained_adapter.json').read_text())['sampler']
    service=tinker.ServiceClient()
    url=service.create_rest_client().get_checkpoint_archive_url_from_tinker_path(ckpt).result().url
    if args.partial and not archive.exists():
        with args.partial.open('rb') as f:
            prefix=f.read(65536)
        with requests.get(url,headers={'Range':f'bytes=0-{len(prefix)-1}'},stream=True,timeout=(30,120)) as response:
            response.raise_for_status()
            assert response.raw.read(len(prefix))==prefix, 'Partial archive does not match checkpoint'
        shutil.copyfile(args.partial,archive)
    offset=archive.stat().st_size if archive.exists() else 0
    headers={'Range':f'bytes={offset}-'} if offset else {}
    with requests.get(url,headers=headers,stream=True,timeout=(30,120)) as response:
        response.raise_for_status()
        if offset:
            assert response.status_code==206
            assert response.headers['Content-Range'].startswith(f'bytes {offset}-')
            total=int(response.headers['Content-Range'].split('/')[-1])
        else:
            total=int(response.headers['Content-Length'])
        print(f'Downloading from {offset:,} of {total:,} bytes',flush=True)
        last=offset
        with archive.open('ab' if offset else 'wb') as f:
            for chunk in response.iter_content(1024*1024):
                f.write(chunk)
                offset+=len(chunk)
                if offset-last>=128*1024*1024:
                    print(f'Downloaded {offset:,}/{total:,} bytes',flush=True)
                    last=offset
        assert archive.stat().st_size==total
    output=out/'adapter'
    output.mkdir(exist_ok=True)
    importlib.import_module(download.__module__)._safe_extract_tar(archive,output)
    print('Archive downloaded and safely extracted',flush=True)


if __name__=='__main__':
    try:
        main()
    except requests.RequestException as exc:
        # Request errors can embed a signed URL; never log its query credentials.
        raise RuntimeError('Checkpoint HTTP transfer failed: '+type(exc).__name__) from None
