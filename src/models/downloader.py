import requests  # type: ignore
import os
from tqdm import tqdm  # type: ignore
from enum import Enum
from concurrent.futures import ThreadPoolExecutor


class Destination(Enum):
    MODEL = 'models'
    WEIGHT = 'weights'


class DownloadParam:
    url: str
    name: str

    @property
    def ext(self) -> str:
        _, ext = os.path.splitext(self.url)
        return ext

    @property
    def path(self) -> str:
        return f'{self.name}{self.ext}'

    def __init__(self, url: str, name: str):
        self.url = url
        self.name = name


def download(param: DownloadParam, dest: Destination):
    destination = dest.value if os.path.dirname(
        dest.value) == '' else os.path.dirname(dest.value)
    os.makedirs(destination, exist_ok=True)

    # ファイルがあればスキップする
    if os.path.exists(f'{destination}/{param.path}'):
        return

    req = requests.get(param.url, stream=True, allow_redirects=True)
    content_length = req.headers.get('content-length')
    progress_bar = tqdm(
        total=int(content_length),
        leave=False,
        unit='B',
        unit_scale=True,
        unit_divisor=1024
    )
    with open(f'{destination}/{param.path}', 'wb') as f:
        for chunk in req.iter_content(chunk_size=1024):
            if chunk:
                progress_bar.update(1024)
                f.write(chunk)


def download_model():
    weights = [
        DownloadParam(
            'https://huggingface.co/ddPn08/rvc-webui-models/resolve/main/embeddings/hubert_base.pt',
            'hubert_base',
        ),
        DownloadParam(
            'https://huggingface.co/rinna/japanese-hubert-base/resolve/main/fairseq/model.pt',
            'hubert_base_jp',
        ),
        DownloadParam(
            'https://huggingface.co/wok000/weights/resolve/main/ddsp-svc30/embedder/hubert-soft-0d54a1f4.pt',
            'hubert_soft',
        ),
        DownloadParam(
            'https://huggingface.co/wok000/weights/resolve/main/ddsp-svc30/nsf_hifigan_20221211/model.bin',
            'nsf_hifigan',
        ),
        DownloadParam(
            'https://huggingface.co/wok000/weights/raw/main/ddsp-svc30/nsf_hifigan_20221211/config.json',
            'config',
        ),
        DownloadParam(
            'https://huggingface.co/wok000/weights/resolve/main/crepe/onnx/full.onnx',
            'crepe_onnx_full'
        ),
        DownloadParam(
            'https://huggingface.co/wok000/weights/resolve/main/crepe/onnx/tiny.onnx',
            'crepe_onnx_tiny'
        ),
        DownloadParam(
            'https://huggingface.co/wok000/weights_gpl/resolve/main/content-vec/contentvec-f.onnx',
            'content_vec_500_onnx'
        ),
        DownloadParam(
            'https://huggingface.co/wok000/weights/resolve/main/rmvpe/rmvpe_20231006.pt',
            'rmvpe'
        ),
        DownloadParam(
            'https://huggingface.co/wok000/weights_gpl/resolve/main/rmvpe/rmvpe_20231006.onnx',
            'rmvpe_onnx'
        ),
        DownloadParam(
            'https://openaipublic.azureedge.net/main/whisper/models/65147644a518d12f04e32d6f3b26facc3f8dd46e5390956a9424a650c0ce22b9/tiny.pt',
            'whisper_tiny'
        ),
    ]
    with ThreadPoolExecutor() as executor:
        for weight in weights:
            executor.submit(download(weight, Destination.WEIGHT))
