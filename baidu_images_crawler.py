#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import itertools
import urllib
import requests
import os
import re
import sys

str_table = {
    '_z2C$q': ':',
    '_z&e3B': '.',
    'AzdH3F': '/'
}

char_table = {
    'w': 'a',
    'k': 'b',
    'v': 'c',
    '1': 'd',
    'j': 'e',
    'u': 'f',
    '2': 'g',
    'i': 'h',
    't': 'i',
    '3': 'j',
    'h': 'k',
    's': 'l',
    '4': 'm',
    'g': 'n',
    '5': 'o',
    'r': 'p',
    'q': 'q',
    '6': 'r',
    'f': 's',
    'p': 't',
    '7': 'u',
    'e': 'v',
    'o': 'w',
    '8': '1',
    'd': '2',
    'n': '3',
    '9': '4',
    'c': '5',
    'm': '6',
    '0': '7',
    'b': '8',
    'l': '9',
    'a': '0'
}

char_table = {ord(key): ord(value) for key, value in char_table.items()}

def decode(url):
    for key, value in str_table.items():
        url = url.replace(key, value)

        return url.translate(char_table)

def buildUrls(word):
    word = urllib.parse.quote(word)
    url = r"http://image.baidu.com/search/acjson?tn=resultjson_com&ipn=rj&ct=201326592&fp=result&queryWord={word}&cl=2&lm=-1&ie=utf-8&oe=utf-8&st=-1&ic=0&word={word}&face=0&istype=2nc=1&pn={pn}&rn=60"
    urls = (url.format(word=word, pn=x) for x in itertools.count(start=0, step=60))
    return urls

re_url = re.compile(r'"objURL":"(.*?)"')
def resolveImgUrl(html):
    imgUrls = [decode(x) for x in re_url.findall(html)]
    return imgUrls

def downImg(imgUrl, dirpath, imgName):
    filename = os.path.join(dirpath, imgName)
    try:
        res = requests.get(imgUrl, timeout=15)
        if str(res.status_code)[0] == "4":
            print(str(res.status_code), ":", imgUrl)
            return False
    except Exception as e:
        print("抛出异常:", imgUrl)
        print(e)
        return False
    with open(filename, "wb") as f:
        f.write(res.content)
    return True

def mkDir(dirName):
    dirpath = os.path.join(sys.path[0], dirName)
    if not os.path.exists(dirpath):
        os.mkdir(dirpath)
    return dirpath

if __name__ == '__main__':
    print("开始下载百度图片")
    print("=" * 50)
    word = input("请输入要下载图片的关键词: \n")
    dirpath = mkDir("results")
    urls = buildUrls(word)
    index = 0
    for url in urls:
        print("请求ing", url)
        html = requests.get(url, timeout=10).content.decode('utf-8')
        imgUrls = resolveImgUrl(html)
        if len(imgUrls) == 0:
            break
        for url in imgUrls:
            if downImg(url, dirpath, str(index) + ".jpg"):
                index += 1
                print("已下载%s张" % index)

