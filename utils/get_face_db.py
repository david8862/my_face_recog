# A simple tool to fetch CRDC TIPBU image from directory with cmd:
#
# $ curl -OL http://wwwin.cisco.com/dir/photo/zoom/xxx.jpg
#

import os

ceclist = [
"denwu",
"choli",
"asuliu",
"xiaowang",
"bolei2",
"chenzhu2",
"zisu",
"junbzhan",
"junhma",
"migu",
"nyuan",
"nzhang2",
"qianx",
"zhzhe",
"siwzhang",
"xiaolihu",
"xiaomma",
"xuaxu",
"yifding",
"yubmao",
"zhixu",
"biychen",
"chanwei",
"cheyang2",
"conli",
"hoqiu",
"feixi",
"jiangfzh",
"jiewa2",
"kaiche",
"rilli",
"xiaobizh",
"hxiaxia",
"zhiholiu",
"zhijin",
"chanwan",
"huijiang",
"shenghli",
"xiaqin",
"xueliang",
"cliu4",
"gtie",
"haniu",
"huigjin",
"zhiyao",
"jianc2",
"jiazhe",
"jibbao",
"jingmxu",
"jujin",
"qzhang2",
"mengl",
"payzhu",
"qiaji",
"lzhang3",
"xzhao3",
"yanmeng",
"yxue2",
"yueyu",
"yuzho2",
"huchen2",
"guozhang",
"hbian",
"junhuang",
"liangxwa",
"lingjin",
"tiqi",
"ycheng3",
"zhpeng",
"haxia",
"gachen2",
"pezhang2",
"ronling",
"shuyzhan",
"xigu",
"yawan",
"allren",
"bruzhang",
"gexue",
"fugd",
"jameshe",
"jiajiang",
"jianzo",
"jiaqshi",
"jingyhua",
"zhoyang",
"xugwu",
"pengzho",
"shafu",
"stevyu",
"xzhuang",
"zhaoli",
"riren",
"jihuo",
"wenloli",
"fanwang2",
"hongdiz",
"huanyliu",
"xuameng",
"peihchen",
"hongdiz",
"shjun",
"wakai",
"shijzhan",
"shugwang",
"taichifu",
"chenyu2",
"cumu",
"dandtang",
"delzhang",
"dongh",
"fangre",
"haisyang",
"haizou",
"hhuiguan",
"jixu3",
"jingalin",
"jiyou",
"jisi",
"leiz3",
"qiujliu",
"milv",
"meig",
"pengfche",
"pzhai",
"piwang2",
"shaliu2",
"shuoy",
"tiren",
"wenca",
"wenlli",
"xiaopzh2",
"xinlxu",
"xixdong",
"xzhuang",
"yany2",
"yanahuan",
"yangw3",
"yanlji",
"yisun2",
"yipzou",
"zhihzhen",
"zhiqizha",
"wenjunl",
"yingpang",
"yuwu2"
]

for name in ceclist:
    os.system("pushd face_db; curl -OL http://wwwin.cisco.com/dir/photo/zoom/{}.jpg; popd".format(name))

