import os

progress = os.getenv('SITATOR_PROGRESSBAR', 'true').lower()
progress = (progress == 'true') or (progress == 'yes') or (progress == 'on')

if progress:
    try:
        from tqdm.autonotebook import tqdm
    except:
        def tqdm(iterable, **kwargs):
            return iterable
else:
    def tqdm(iterable, **kwargs):
        return iterable
