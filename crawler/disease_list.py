"""预定义疾病列表 —— 爬虫默认目标（确保首次运行即可爬到足量数据）。

格式：(疾病名, 好大夫词条 slug)。详情页 URL = f"{BASE}/citiao/jibing-{slug}.html"
覆盖心血管、呼吸、消化、内分泌、神经、骨科、妇产、儿科、皮肤、五官等主要科室。
爬虫礼仪：爬取前先过 robots.txt 检查（medical_crawler.can_fetch）。
"""

# (name, slug)
PREDEFINED_DISEASES: list[tuple[str, str]] = [
    # 心血管 / 内分泌
    ("高血压", "gaoxueya"), ("冠心病", "guanxinbing"), ("心律失常", "xinlvshichang"),
    ("心力衰竭", "xinlishuaijie"), ("心肌缺血", "xinjiquexue"), ("心肌炎", "xinjiyan"),
    ("心绞痛", "xinjiaotong"), ("高血脂", "gaoxuezhi"), ("动脉硬化", "dongmaiyinghua"),
    ("低血压", "dixueya"), ("糖尿病", "tangniaobing"), ("甲亢", "jiakang"),
    ("甲减", "jiajian"), ("痛风", "tongfeng"), ("肥胖症", "feipangzheng"),
    ("骨质疏松", "guzhishusong"), ("高尿酸血症", "gaoniaosuanxuezheng"),
    # 呼吸
    ("感冒", "ganmao"), ("流感", "liugan"), ("肺炎", "feiyan"), ("哮喘", "xiaochuan"),
    ("慢性支气管炎", "manxingzhiqiguanyan"), ("肺气肿", "feiqizhong"),
    ("支气管扩张", "zhiqiguankuozhang"), ("肺结核", "feijiehe"), ("肺癌", "feiai"),
    ("肺心病", "feixinbing"), ("扁桃体炎", "biantaotiyan"), ("鼻窦炎", "bidouyan"),
    ("过敏性鼻炎", "guominxingbiyan"),
    # 消化
    ("胃炎", "weiyan"), ("胃溃疡", "weikuiyang"), ("十二指肠溃疡", "shierzhichangkuiyang"),
    ("胃癌", "weiai"), ("肝炎", "ganyan"), ("肝硬化", "ganyinghua"),
    ("脂肪肝", "zhifanggan"), ("胆囊炎", "dannangyan"), ("胆结石", "danjieshi"),
    ("胰腺炎", "yixianyan"), ("肠炎", "changyan"), ("阑尾炎", "lanweiyan"),
    ("痔疮", "zhichuang"), ("便秘", "bianmi"), ("腹泻", "fuxie"),
    ("肠易激综合征", "changyijizonghezheng"), ("食管炎", "shiguanyan"), ("食管癌", "shiguanai"),
    ("结直肠癌", "jiezhichangai"), ("消化不良", "xiaohuabuliang"),
    # 神经
    ("头痛", "toutong"), ("偏头痛", "piantoutong"), ("失眠", "shimian"),
    ("脑梗塞", "naogengse"), ("脑出血", "naochuxue"), ("癫痫", "dianxian"),
    ("帕金森病", "pajinsenbing"), ("老年痴呆", "laonianchidai"), ("面瘫", "miantan"),
    ("三叉神经痛", "sanchashenjingtong"), ("坐骨神经痛", "zuogushenjingtong"),
    ("眩晕症", "xuanyunzheng"), ("神经衰弱", "shenjingshuairuo"),
    # 骨科 / 外科
    ("颈椎病", "jingzhuibing"), ("腰椎间盘突出", "yaozhijiantuchu"),
    ("肩周炎", "jianzhouyan"), ("关节炎", "guanjieyan"), ("类风湿关节炎", "leifengshiguanjieyan"),
    ("骨折", "guzhe"), ("腱鞘炎", "jianqiaoyan"), ("股骨头坏死", "guutouhuaisi"),
    ("强直性脊柱炎", "qiangzhixingjizhuyan"), ("滑膜炎", "huamoyan"), ("骨质增生", "guzhizengsheng"),
    ("落枕", "luozhen"), ("阑尾癌", "lanweiai"), ("静脉曲张", "jingmaiquzhang"),
    # 泌尿 / 肾内科
    ("肾炎", "shenyan"), ("肾结石", "shenjieshi"), ("尿路感染", "niaolugangan"),
    ("前列腺炎", "qianliexianyan"), ("前列腺增生", "qianliexianzengsheng"),
    ("肾功能不全", "shengongnengbuquan"), ("膀胱炎", "pangguangyan"),
    ("泌尿系结石", "miniaoxijieshi"), ("阳痿", "yangwei"), ("早泄", "zaoxie"),
    # 妇产
    ("月经不调", "yuejingbutiao"), ("痛经", "tongjing"), ("阴道炎", "yindao yan".replace(" ", "")),
    ("宫颈炎", "gongjingyan"), ("子宫肌瘤", "zigongjiliu"), ("盆腔炎", "penqiangyan"),
    ("多囊卵巢综合征", "duonangluanchaozonghezheng"), ("不孕不育", "buyunbuyu"),
    ("妊娠糖尿病", "renshentangniaobing"), ("更年期综合征", "gengnianqizonghezheng"),
    ("乳腺增生", "ruxianzengsheng"), ("乳腺癌", "ruxianai"), ("宫颈糜烂", "gongjingmilan"),
    # 儿科
    ("小儿感冒", "xiaoerganmao"), ("小儿咳嗽", "xiaoerkesou"), ("小儿肺炎", "xiaoerfeiyan"),
    ("小儿腹泻", "xiaoerfuxie"), ("手足口病", "shouzukoubing"), ("水痘", "shuidou"),
    ("麻疹", "mazhen"), ("风疹", "fengzhen"), ("猩红热", "xinghongre"),
    ("小儿哮喘", "xiaoerxiaochuan"), ("新生儿黄疸", "xinshengerhuangdan"),
    # 皮肤
    ("湿疹", "shizhen"), ("荨麻疹", "xunmazhen"), ("痤疮", "cuochuang"),
    ("银屑病", "yinxiebing"), ("白癜风", "baidianfeng"), ("带状疱疹", "daizhuangpaozhen"),
    ("手足癣", "shouzouxuan"), ("脱发", "tuofa"), ("狐臭", "huchou"), ("冻疮", "dongchuang"),
    # 五官 / 眼科 / 口腔
    ("结膜炎", "jiemoyan"), ("角膜炎", "jiaomoyan"), ("白内障", "baineizhang"),
    ("青光眼", "qingguangyan"), ("干眼症", "ganyanzheng"), ("近视", "jinshi"),
    ("中耳炎", "zhongeryan"), ("耳鸣", "erming"), ("咽炎", "yanyan"),
    ("口腔溃疡", "kouqiangkuiyang"), ("牙周炎", "yazhouyan"), ("龋齿", "quchi"),
    # 精神心理 / 血液 / 其他
    ("抑郁症", "yiyuzheng"), ("焦虑症", "jiaolüzheng".replace("ü", "lv")),
    ("神经官能症", "shenjingguannengzheng"), ("贫血", "pinxue"),
    ("白血病", "baixuebing"), ("血小板减少", "xuexiaobanjianshao"),
    ("过敏性紫癜", "guominxingzidian"), ("中暑", "zhongshu"), ("铅中毒", "qianzhongdu"),
    ("食物中毒", "shiwuzhongdu"), ("酒精肝", "jiujinggan"), ("药物性肝损伤", "yaowuxinggansunshang"),
    ("亚健康", "yajiankang"), ("体癣", "tixuan"), ("灰指甲", "huizhijia"),
    ("青春痘", "qingchundou"), ("黄褐斑", "huangheban"), ("癌症疼痛", "aizhengtengtong"),
]


def get_disease_url(base_url: str, slug: str) -> str:
    return f"{base_url}/citiao/jibing-{slug}.html"
