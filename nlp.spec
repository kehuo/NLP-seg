# -*- mode: python -*-
import os
tensorflow_binaries = []
tensorflow_location = '/root/.pyenv/versions/py3_env/lib/python3.6/site-packages/tensorflow'

for dir_name, sub_dir_list, fileList in os.walk(tensorflow_location):
  for file in fileList:
    if file.endswith(".so"):
      full_file = dir_name + '/' + file
      print(full_file)
      tensorflow_binaries.append((full_file, '.'))

block_cipher = None

a = Analysis(['nlp.py'],
             pathex=['/project/NLP-seg'],
             binaries=tensorflow_binaries,
             datas=[('./nlp_seg/dict/BBMEDI.dat', './nlp_seg/dict')],
             hiddenimports=['gunicorn.workers.ggevent', 'sklearn.utils._cython_blas'],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          a.binaries,
          a.zipfiles,
          a.datas,
          [],
          name='nlp',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          runtime_tmpdir=None,
          console=True )
