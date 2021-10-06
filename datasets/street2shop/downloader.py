## Based on: https://github.com/pumpikano/street2shop, adapted for Py3
import imghdr
import os
import time
from multiprocessing import Pool
from multiprocessing.dummy import freeze_support
from pathlib import Path
from urllib.parse import urlparse

import requests
from tqdm import tqdm

from default_logger.defaultLogger import defaultLogger
from scrapper.util.list import distinct

THREADS = 32
RETRIES = 36

link_path = f"F:\workspace\datasets\street2shop\photos.txt"
base_path = f"D:\dl_target"
image_path = Path(base_path) / "imgs"

logger = defaultLogger("Street2Shop Downloader")

if not Path(link_path).exists():
    raise Exception(f"Link Path ({link_path}) does not exist!")


def load_id_url(path: str, force=False):
    clean_line = lambda l: l.replace("\n", " ").strip()
    images = os.listdir(image_path) if not force else None

    def split_line(l):
        first_sep = l.find(",")
        return l[:first_sep], l[first_sep + 1:]

    def pop_filter(l):
        """ images and lines are sorted -> should be close to O(1) """
        try:
            idx = images.index(l[1])
            del images[idx]
            return False
        except ValueError:
            return True

    def remove_ext(f_name):
        return f_name.split(".")[0]

    with open(path) as f:
        lines = f.readlines()
        lines = map(clean_line, lines)
        id_url = list(map(split_line, lines))

        if not force:
            images = list(map(remove_ext, images))
            pre_len = len(id_url)
            id_url_iter = tqdm(id_url, desc="Filter (already) downloaded Images")
            id_url = list(filter(pop_filter, id_url_iter))
            after_len = len(id_url)
            logger.debug(f"DL Links. ({pre_len}/{after_len} = {pre_len / after_len}%) already downloaded.")

        return id_url


def download(id_url):
    try:
        id_, url = id_url
        r = requests.get(url)

        if r.ok:
            img_type = imghdr.what(None, r.content)

            if img_type is not None:
                with open(os.path.join(image_path, id_ + '.' + img_type), 'wb') as f:
                    f.write(r.content)
                    f.close()
                    return None
            else:
                logger.debug(f"Unknown Type {id_}, {url}")
        else:
            logger.debug(f"{url}: {r.status_code}")
    except Exception as e:
        pass

    return id_, url


if __name__ == '__main__':
    freeze_support()

    image_path.mkdir(parents=True, exist_ok=True)
    logger.debug(f"DST: {image_path}")

    dl_jobs = load_id_url(link_path)

    last_job_len = len(dl_jobs)

    for run_idx in reversed(range(RETRIES)):
        threads = THREADS + run_idx - RETRIES  #<- Run 1: T Threads, Run 2: T-2 Threads, Run N: T-N
        threads = max(threads, 2)              # <- Min 2 Threads
        total = len(dl_jobs)
        logger.debug(f"Prev. len(JOB): {last_job_len}, Current: {total} -> Dif: {last_job_len-total}")

        with Pool(threads) as p:
            p_itter = p.imap(download, dl_jobs)
            dl_jobs = list(tqdm(p_itter, total=total, desc=f"{threads} threads. Run: {run_idx}"))
            dl_jobs = list(filter(lambda x: x is not None, dl_jobs))

        if len(dl_jobs) < 1:
            logger.debug("DONE")
            break

        last_job_len = len(dl_jobs)
        time.sleep(30)
    failed_file_path = Path(base_path) / "failed.txt"
    logger.debug(f"{failed_file_path}")

#    with open(failed_file_path, "w") as f:
#        lines = map(lambda l: ",".join(l), dl_jobs)
#        f.writelines(lines)
