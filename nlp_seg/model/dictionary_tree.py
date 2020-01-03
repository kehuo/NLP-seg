#!/usr/bin/python
# coding=utf-8
import base64
import pkgutil
import codecs
from Crypto.Cipher import AES
from Crypto.Hash import MD5
from binascii import a2b_hex, b2a_hex

text = 'Ds7#ds9s7$kascv4'

def align(str, isKey=False):
    if isKey:
        if len(str) > 16:
            return str[0:16]
        else:
            return align(str)
    else:
        zerocount = 16-len(str) % 16
        for i in range(0, zerocount):
            str = str + '\0'
        return str


def decrypt_CBC(str, key):
    key = align(key, True)
    try:
        AESCipher = AES.new(key, AES.MODE_CBC, '7261926512581232')
    except:
        AESCipher = AES.new(key.encode(), AES.MODE_CBC, '7261926512581232'.encode())
    paint = AESCipher.decrypt(a2b_hex(str))
    return paint


def load_dat(dict_file, key=text):
    lines = []
    content = codecs.open(dict_file, 'r', encoding='utf-8').read()
    # content = pkgutil.get_data(module, dict_file).decode('UTF-8', 'ignore')

    fcipherText = base64.b64decode(content).decode('utf8')
    if len(fcipherText.split(',')) < 2:
        return lines
    cipherText = fcipherText.split(',')[0]
    painhash = fcipherText.split(',')[1]
    painText = decrypt_CBC(cipherText, key).decode('utf8')
    painText = painText.rstrip('\0')
    MD5hash = MD5.new()
    MD5hash.update(painText.encode())
    if painhash == MD5hash.hexdigest():
        raw = base64.b64decode(painText.encode()).decode('utf8')
        lines = raw.split('\n')
    return lines


class DictionaryTree(object):
    def __init__(self, noConflict=True):
        self.noConflict = noConflict
        return

    def loadDicFileItems(self, dicPath):
        fd = open(dicPath, 'r')
        items = []
        while True:
            line = fd.readline()
            if line == '':
                break
            fields = line.strip().split('\t')
            if len(fields)<2:
                continue
            # type, text
            items.append((fields[0], fields[1]))
        fd.close()
        return items

    def loadDicFileItemsfromDAT(self, dict_file):
        items = []
        lines = load_dat(dict_file)
        for line in lines:
            if line == '':
                break
            fields = line.strip().split('\t')
            if len(fields) < 2:
                continue
            # type, text
            items.append((fields[0], fields[1]))
        return items

    def buildGoDeep(self, dicTree, descriptionText, tagType):
        curTree = dicTree
        curText = ''
        conflict = False
        for charOne in descriptionText:
            curText = curText + charOne
            if charOne in curTree:
                # go deep for sub tree
                curTree = curTree[charOne]
                if self.noConflict and 'tag' in curTree:
                    # conflict with somebody
                    conflict = True
                    break
            else:
                # build new sub tree
                curTree[charOne] = {}
                curTree = curTree[charOne]
        if conflict:
           return
        if self.noConflict and len(curTree)>0:
            # curText = descriptionText + json.dumps(curTree, ensure_ascii=False)
            # print('--- node as partial", tagType, curText, descriptionText')
            return
        if 'tag' in curTree:
            # print('--- tag already(replaced)', tagType, curTree['tag'])
            pass
        curTree['tag'] = tagType
        return

    def buildTree(self, items):
        dicTree = {}
        for itemOne in items:
            tagType = itemOne[0]
            descriptionText = itemOne[1]
            self.buildGoDeep(dicTree, descriptionText, tagType)
        return dicTree

    def load(self, dict_file):
        items = self.loadDicFileItemsfromDAT(dict_file)
        dicTree = self.buildTree(items)
        self.dicTree = dicTree
        return

    def searchInit(self):
        self.candidates = []
        return

    def searchOneChar(self, charOne, curIdx):
        # print('candidates length=', len(self.candidates))

        # add a new start instance {curTree,start}
        newBorn = {
            'curTree':self.dicTree, 'start':curIdx
        }
        self.candidates.append(newBorn)

        rst = []
        newCandidates = []
        for workerOne in self.candidates:
            curTree = workerOne['curTree']
            if not charOne in curTree:
                # no word path be found anymore, abandon this candidate
                continue
            curTree = curTree[charOne]
            if 'tag' in curTree:
                # here, a word found and abandon this candidate
                rst.append([workerOne['start'], curIdx, curTree['tag']])
                if self.noConflict:
                    continue
            # some word path still can be tracked, go deep and add as new candidate
            workerOne['curTree'] = curTree
            newCandidates.append(workerOne)
        self.candidates = newCandidates
        return rst

    def removeConflictCandidates(self, resultTags):
        if len(resultTags) == 0:
            return []
        resultTags.sort(key=lambda t: t[0])
        rst = []
        curStartIdx = 0
        curEndIdx = -1
        curTagType = None
        for tagOne in resultTags:
            tagStart = tagOne[0]
            tagEnd = tagOne[1]
            tagType = tagOne[2]
            operation = None
            if tagStart > curEndIdx:
                # a separated new tag found, flush previous one first
                operation = 'flush'
            elif tagStart == curEndIdx:
                # check tag length first, and keep longer one
                if curEndIdx-curStartIdx < tagEnd-tagStart:
                    operation = 'replace'
                else:
                    operation = 'skip'
            # ---here tagStart < curEndIdx
            elif tagEnd > curEndIdx:
                # longer match tag be found, should not happen because of strict dictionary build process
                # todo test
                if curEndIdx-curStartIdx < tagEnd-tagStart:
                    operation = 'replace'
                else:
                    operation = 'skip'
            else:
                # here, tagEnd<=curEndIdx and tagStart>=curStartIdx
                # anyway, just skip this short candidate
                operation = 'skip'

            if operation == 'replace':
                curStartIdx = tagStart
                curEndIdx = tagEnd
                curTagType = tagType
            elif operation == 'flush':
                if not curTagType is None:
                    rst.append([curStartIdx, curEndIdx, curTagType])
                curStartIdx = tagStart
                curEndIdx = tagEnd
                curTagType = tagType
            elif operation == 'skip':
                pass
        # last flush if possible
        if not curTagType is None:
            rst.append([curStartIdx, curEndIdx, curTagType])
        return rst

    def search(self, content):
        self.searchInit()
        resultTags = []
        curIdx = 0
        for charOne in content:
            curTags = self.searchOneChar(charOne, curIdx)
            if len(curTags)>0:
                resultTags.extend(curTags)
            curIdx = curIdx + 1
        rst = self.removeConflictCandidates(resultTags)
        return rst

def doTestDictionaryTree(cfg):
    dicPath = cfg['input']
    noConflict = False
    if 'no_conflict' in cfg:
        noConflict = (cfg['no_conflict']=='1')
    dicTree = DictionaryTree(noConflict)
    dicTree.load(dicPath)

    content = cfg['content']
    resultTags = dicTree.search(content)

    for tagOne in resultTags:
        # start, end, tag
        start = tagOne[0]
        end = tagOne[1]
    return

