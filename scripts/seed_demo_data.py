"""种子演示数据生成器 —— 离线跑通全链路（无需爬虫/API Key）。

产出与真实管线完全同构（同格式同契约，R3/R5）：
  data/processed/diseases.json          （爬虫清洗产物格式）
  data/knowledge_graph/entities.json    （抽取产物格式）
  data/knowledge_graph/relations.json   （抽取产物格式）

真实数据路径（有网络/Key 时）：medical_crawler → data_processor → run_extraction。
本脚本仅用于开发/演示与自动化测试。

用法：python scripts/seed_demo_data.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.paths import (
    KG_ENTITIES_FILE,
    KG_RELATIONS_FILE,
    PROCESSED_DISEASES_JSON,
)
from src.common.utils import write_json

# ---------------------------------------------------------------------------
# 40 种常见疾病的策划知识（症状/科室/药物/检查/身体部位/预防/饮食），够撑起
# 节点数 >100、密度 >0.5 的演示级知识图谱。数据为公开医学科普常识。
# ---------------------------------------------------------------------------
SEED: dict[str, dict] = {
    "高血压": {
        "description": "以体循环动脉血压增高为主要特征，可伴有心、脑、肾等器官的功能或器质性损害的临床综合征。",
        "symptoms": ["头痛", "头晕", "心悸", "耳鸣", "疲劳"],
        "department": "心血管内科",
        "medications": ["氨氯地平", "缬沙坦", "美托洛尔", "氢氯噻嗪"],
        "examinations": ["血压测量", "心电图", "血脂检查", "肾功能检查"],
        "body_parts": ["心脏", "血管", "肾脏"],
        "precautions": ["低盐饮食", "规律监测血压", "避免熬夜", "适量运动"],
        "foods": ["低盐饮食", "芹菜", "香蕉"], "avoid_foods": ["腌制食品", "高脂饮食"],
        "category": {"氨氯地平": "钙通道阻滞剂", "缬沙坦": "血管紧张素II受体拮抗剂",
                     "美托洛尔": "β受体阻滞剂", "氢氯噻嗪": "利尿剂"},
    },
    "糖尿病": {
        "description": "以高血糖为特征的代谢性疾病，由胰岛素分泌缺陷或其生物作用受损引起。",
        "symptoms": ["多饮", "多尿", "多食", "体重下降", "乏力"],
        "department": "内分泌科",
        "medications": ["二甲双胍", "格列美脲", "胰岛素"],
        "examinations": ["血糖检测", "糖化血红蛋白", "尿常规"],
        "body_parts": ["胰腺", "血管"],
        "precautions": ["控制饮食", "规律监测血糖", "适量运动", "定期复查"],
        "foods": ["粗粮", "绿叶蔬菜"], "avoid_foods": ["高糖食品", "含糖饮料"],
        "category": {"二甲双胍": "双胍类降糖药", "格列美脲": "磺脲类降糖药", "胰岛素": "降糖生物制剂"},
    },
    "冠心病": {
        "description": "冠状动脉粥样硬化使血管狭窄或阻塞，导致心肌缺血缺氧或坏死的心脏病。",
        "symptoms": ["胸痛", "心悸", "胸闷", "气短"],
        "department": "心血管内科",
        "medications": ["阿司匹林", "阿托伐他汀", "硝酸甘油"],
        "examinations": ["心电图", "冠状动脉造影", "心脏彩超"],
        "body_parts": ["心脏", "冠状动脉"],
        "precautions": ["避免劳累", "随身携带急救药物", "戒烟限酒"],
        "foods": ["低脂饮食", "深海鱼"], "avoid_foods": ["高脂饮食", "动物内脏"],
        "category": {"阿司匹林": "抗血小板药", "阿托伐他汀": "他汀类调脂药", "硝酸甘油": "硝酸酯类"},
    },
    "感冒": {
        "description": "上呼吸道感染，由病毒引起的常见急性呼吸道疾病。",
        "symptoms": ["鼻塞", "流涕", "咽痛", "咳嗽", "发热"],
        "department": "呼吸内科",
        "medications": ["对乙酰氨基酚", "感冒灵颗粒"],
        "examinations": ["血常规"],
        "body_parts": ["鼻", "咽喉", "上呼吸道"],
        "precautions": ["多喝水", "注意休息", "注意保暖"],
        "foods": ["热水", "清淡饮食"], "avoid_foods": ["辛辣刺激食物"],
        "category": {"对乙酰氨基酚": "解热镇痛药", "感冒灵颗粒": "复方感冒药"},
    },
    "流感": {
        "description": "由流感病毒引起的急性呼吸道传染病，传染性强，症状较普通感冒重。",
        "symptoms": ["高热", "头痛", "乏力", "肌肉酸痛", "咳嗽"],
        "department": "呼吸内科",
        "medications": ["奥司他韦", "对乙酰氨基酚"],
        "examinations": ["血常规", "流感病毒抗原检测"],
        "body_parts": ["呼吸道", "肺"],
        "precautions": ["隔离休息", "戴口罩", "勤洗手"],
        "foods": ["温水", "易消化食物"], "avoid_foods": ["油腻食物"],
        "category": {"奥司他韦": "抗病毒药", "对乙酰氨基酚": "解热镇痛药"},
    },
    "肺炎": {
        "description": "肺泡、远端气道和肺间质的感染性炎症，可由细菌、病毒等病原体引起。",
        "symptoms": ["发热", "咳嗽", "咳痰", "胸痛", "呼吸困难"],
        "department": "呼吸内科",
        "medications": ["阿莫西林", "阿奇霉素"],
        "examinations": ["胸部X光", "胸部CT", "血常规", "痰培养"],
        "body_parts": ["肺", "支气管"],
        "precautions": ["及时就医", "卧床休息", "保持室内通风"],
        "foods": ["温热流食"], "avoid_foods": ["生冷食物"],
        "category": {"阿莫西林": "青霉素类抗生素", "阿奇霉素": "大环内酯类抗生素"},
    },
    "哮喘": {
        "description": "气道慢性炎症性疾病，表现为反复发作的喘息、气促、胸闷和咳嗽。",
        "symptoms": ["喘息", "气促", "胸闷", "咳嗽"],
        "department": "呼吸内科",
        "medications": ["布地奈德", "沙丁胺醇"],
        "examinations": ["肺功能检查", "过敏原检测"],
        "body_parts": ["支气管", "肺"],
        "precautions": ["远离过敏原", "随身携带吸入剂", "避免剧烈运动"],
        "foods": [], "avoid_foods": ["已知过敏食物"],
        "category": {"布地奈德": "吸入性糖皮质激素", "沙丁胺醇": "β2受体激动剂"},
    },
    "慢性支气管炎": {
        "description": "气管、支气管黏膜及其周围组织的慢性非特异性炎症，咳痰喘每年持续3个月。",
        "symptoms": ["咳嗽", "咳痰", "气喘"],
        "department": "呼吸内科",
        "medications": ["氨溴索", "茶碱"],
        "examinations": ["胸部X光", "肺功能检查"],
        "body_parts": ["支气管", "气管"],
        "precautions": ["戒烟", "防寒保暖", "预防感冒"],
        "foods": ["白萝卜", "梨"], "avoid_foods": ["烟草", "辛辣食物"],
        "category": {"氨溴索": "祛痰药", "茶碱": "支气管扩张剂"},
    },
    "胃炎": {
        "description": "胃黏膜对胃内各种刺激因素的炎症反应，分为急性与慢性。",
        "symptoms": ["上腹痛", "腹胀", "恶心", "嗳气", "反酸"],
        "department": "消化内科",
        "medications": ["奥美拉唑", "铝碳酸镁", "阿莫西林"],
        "examinations": ["胃镜", "幽门螺杆菌检测"],
        "body_parts": ["胃"],
        "precautions": ["规律饮食", "避免刺激性食物", "戒烟酒"],
        "foods": ["小米粥", "面条"], "avoid_foods": ["辛辣食物", "酒精", "浓茶咖啡"],
        "category": {"奥美拉唑": "质子泵抑制剂", "铝碳酸镁": "胃黏膜保护剂", "阿莫西林": "青霉素类抗生素"},
    },
    "胃溃疡": {
        "description": "胃黏膜被胃酸和胃蛋白酶消化形成的溃疡，餐后痛为典型表现。",
        "symptoms": ["上腹痛", "反酸", "嗳气", "恶心"],
        "department": "消化内科",
        "medications": ["奥美拉唑", "克拉霉素", "铝碳酸镁"],
        "examinations": ["胃镜", "幽门螺杆菌检测"],
        "body_parts": ["胃"],
        "precautions": ["规律进餐", "避免空腹刺激", "遵医嘱根除幽门螺杆菌"],
        "foods": ["易消化食物"], "avoid_foods": ["酒精", "辛辣食物", "阿司匹林等刺激性药物"],
        "category": {"奥美拉唑": "质子泵抑制剂", "克拉霉素": "大环内酯类抗生素", "铝碳酸镁": "胃黏膜保护剂"},
    },
    "腹泻": {
        "description": "排便次数明显超过平日习惯，粪质稀薄，水分增加的常见消化道症状。",
        "symptoms": ["腹痛", "稀便", "里急后重", "发热"],
        "department": "消化内科",
        "medications": ["蒙脱石散", "口服补液盐", "诺氟沙星"],
        "examinations": ["大便常规", "血常规"],
        "body_parts": ["肠道", "胃"],
        "precautions": ["补充水分", "注意饮食卫生", "腹部保暖"],
        "foods": ["米汤", "面条", "口服补液盐"], "avoid_foods": ["生冷食物", "油腻食物"],
        "category": {"蒙脱石散": "止泻吸附剂", "口服补液盐": "电解质补充剂", "诺氟沙星": "喹诺酮类抗生素"},
    },
    "便秘": {
        "description": "排便次数减少、粪便干硬和（或）排便困难的消化道常见症状。",
        "symptoms": ["排便困难", "腹胀", "腹痛"],
        "department": "消化内科",
        "medications": ["乳果糖", "开塞露"],
        "examinations": ["便常规", "肠镜"],
        "body_parts": ["肠道", "直肠"],
        "precautions": ["多摄入膳食纤维", "养成定时排便习惯", "适量运动"],
        "foods": ["香蕉", "燕麦", "酸奶"], "avoid_foods": ["辛辣食物"],
        "category": {"乳果糖": "渗透性泻药", "开塞露": "刺激性泻药"},
    },
    "脂肪肝": {
        "description": "肝细胞内脂肪堆积过多的病变，与肥胖、饮酒、代谢异常密切相关。",
        "symptoms": ["乏力", "右上腹不适", "肝区隐痛"],
        "department": "消化内科",
        "medications": ["水飞蓟素"],
        "examinations": ["肝脏B超", "肝功能检查", "血脂检查"],
        "body_parts": ["肝"],
        "precautions": ["控制体重", "戒酒", "规律运动"],
        "foods": ["蔬菜", "优质蛋白"], "avoid_foods": ["高脂饮食", "酒精", "含糖饮料"],
        "category": {"水飞蓟素": "保肝药"},
    },
    "头痛": {
        "description": "头颅上半部疼痛的常见症状，病因多样，从紧张性头痛到器质性疾病。",
        "symptoms": ["头部胀痛", "搏动性疼痛", "头晕"],
        "department": "神经内科",
        "medications": ["布洛芬", "对乙酰氨基酚"],
        "examinations": ["头部CT", "脑电图"],
        "body_parts": ["头部"],
        "precautions": ["规律作息", "避免熬夜", "减少压力"],
        "foods": [], "avoid_foods": ["酒精", "含酪胺食物"],
        "category": {"布洛芬": "非甾体抗炎药", "对乙酰氨基酚": "解热镇痛药"},
    },
    "偏头痛": {
        "description": "反复发作的搏动性头痛，多为一侧，常伴恶心、畏光畏声。",
        "symptoms": ["单侧头痛", "恶心", "畏光", "畏声"],
        "department": "神经内科",
        "medications": ["布洛芬", "舒马曲普坦"],
        "examinations": ["头部CT", "脑电图"],
        "body_parts": ["头部", "血管"],
        "precautions": ["记录头痛日记", "避免诱因", "保证睡眠"],
        "foods": ["镁丰富食物"], "avoid_foods": ["红酒", "奶酪", "巧克力"],
        "category": {"布洛芬": "非甾体抗炎药", "舒马曲普坦": "曲普坦类特异性药物"},
    },
    "失眠": {
        "description": "无法入睡或保持睡眠状态，导致睡眠不足的常见睡眠障碍。",
        "symptoms": ["入睡困难", "早醒", "白天嗜睡", "注意力下降"],
        "department": "神经内科",
        "medications": ["佐匹克隆", "枣仁安神胶囊"],
        "examinations": ["多导睡眠监测"],
        "body_parts": ["大脑"],
        "precautions": ["规律作息", "睡前远离手机", "避免午后咖啡因"],
        "foods": ["牛奶", "小米"], "avoid_foods": ["咖啡", "浓茶", "酒精"],
        "category": {"佐匹克隆": "镇静催眠药", "枣仁安神胶囊": "中成药"},
    },
    "脑梗塞": {
        "description": "脑部血液供应障碍，缺血缺氧导致的局限性脑组织缺血性坏死或软化。",
        "symptoms": ["肢体无力", "口角歪斜", "言语不清", "头晕"],
        "department": "神经内科",
        "medications": ["阿司匹林", "阿托伐他汀", "氯吡格雷"],
        "examinations": ["头部CT", "头部核磁共振", "颈动脉超声"],
        "body_parts": ["大脑", "脑血管"],
        "precautions": ["控制三高", "紧急情况立即拨打120", "康复训练"],
        "foods": ["低盐低脂饮食"], "avoid_foods": ["高脂饮食", "高盐饮食"],
        "category": {"阿司匹林": "抗血小板药", "阿托伐他汀": "他汀类调脂药", "氯吡格雷": "抗血小板药"},
    },
    "颈椎病": {
        "description": "颈椎间盘退行性变及其继发病理改变，刺激或压迫相邻组织引起症状。",
        "symptoms": ["颈肩痛", "上肢麻木", "头晕", "头痛"],
        "department": "骨科",
        "medications": ["塞来昔布", "甲钴胺"],
        "examinations": ["颈椎X光", "颈椎核磁共振"],
        "body_parts": ["颈椎", "脊髓"],
        "precautions": ["避免长时间低头", "颈部保暖", "适当锻炼颈肩肌肉"],
        "foods": [], "avoid_foods": [],
        "category": {"塞来昔布": "非甾体抗炎药", "甲钴胺": "神经营养药"},
    },
    "腰椎间盘突出": {
        "description": "腰椎间盘髓核突出压迫神经根，引起腰痛和下肢放射痛。",
        "symptoms": ["腰痛", "下肢放射痛", "麻木", "活动受限"],
        "department": "骨科",
        "medications": ["塞来昔布", "甲钴胺", "乙哌立松"],
        "examinations": ["腰椎核磁共振", "腰椎CT"],
        "body_parts": ["腰椎", "脊髓", "坐骨神经"],
        "precautions": ["避免久坐", "睡硬板床", "避免弯腰搬重物"],
        "foods": [], "avoid_foods": [],
        "category": {"塞来昔布": "非甾体抗炎药", "甲钴胺": "神经营养药", "乙哌立松": "肌松药"},
    },
    "类风湿关节炎": {
        "description": "以侵蚀性关节炎为主要表现的自身免疫病，晨僵为典型特征。",
        "symptoms": ["关节肿痛", "晨僵", "关节畸形", "乏力"],
        "department": "风湿免疫科",
        "medications": ["甲氨蝶呤", "来氟米特"],
        "examinations": ["类风湿因子", "抗CCP抗体", "关节X光"],
        "body_parts": ["关节", "滑膜"],
        "precautions": ["注意关节保暖", "适度功能锻炼", "定期复查"],
        "foods": ["富含Omega-3食物"], "avoid_foods": ["高嘌呤食物"],
        "category": {"甲氨蝶呤": "抗风湿改善病情药", "来氟米特": "抗风湿改善病情药"},
    },
    "痛风": {
        "description": "嘌呤代谢紊乱和（或）尿酸排泄减少所致的高尿酸血症直接相关的代谢性疾病。",
        "symptoms": ["关节剧痛", "关节红肿", "发热"],
        "department": "内分泌科",
        "medications": ["秋水仙碱", "别嘌醇", "非布司他"],
        "examinations": ["血尿酸检测", "关节超声"],
        "body_parts": ["关节", "肾脏"],
        "precautions": ["低嘌呤饮食", "多喝水", "戒酒"],
        "foods": ["低脂奶制品", "樱桃"], "avoid_foods": ["动物内脏", "海鲜", "啤酒", "肉汤"],
        "category": {"秋水仙碱": "抗痛风急性发作药", "别嘌醇": "抑制尿酸生成药", "非布司他": "抑制尿酸生成药"},
    },
    "贫血": {
        "description": "人体外周血红细胞容量减少，低于正常范围下限的常见临床症状。",
        "symptoms": ["乏力", "头晕", "面色苍白", "心悸"],
        "department": "血液科",
        "medications": ["硫酸亚铁", "维生素B12"],
        "examinations": ["血常规", "铁代谢检查"],
        "body_parts": ["血液", "骨髓"],
        "precautions": ["均衡营养", "查明病因", "经期女性注意补铁"],
        "foods": ["瘦肉", "动物肝脏", "菠菜"], "avoid_foods": ["浓茶", "咖啡"],
        "category": {"硫酸亚铁": "铁剂", "维生素B12": "维生素类"},
    },
    "过敏性鼻炎": {
        "description": "接触过敏原后由IgE介导的鼻黏膜非感染性慢性炎症性疾病。",
        "symptoms": ["鼻痒", "阵发性喷嚏", "清水样鼻涕", "鼻塞"],
        "department": "耳鼻喉科",
        "medications": ["氯雷他定", "糠酸莫米松鼻喷雾剂"],
        "examinations": ["过敏原检测", "鼻内镜"],
        "body_parts": ["鼻"],
        "precautions": ["远离过敏原", "佩戴口罩", "鼻腔冲洗"],
        "foods": [], "avoid_foods": ["已知过敏食物"],
        "category": {"氯雷他定": "抗组胺药", "糠酸莫米松鼻喷雾剂": "鼻用糖皮质激素"},
    },
    "结膜炎": {
        "description": "结膜受病原体感染或刺激发生的炎症，俗称红眼病（传染性者）。",
        "symptoms": ["眼红", "眼痒", "异物感", "分泌物增多"],
        "department": "眼科",
        "medications": ["左氧氟沙星滴眼液", "色甘酸钠滴眼液"],
        "examinations": ["裂隙灯检查"],
        "body_parts": ["眼"],
        "precautions": ["不用手揉眼", "毛巾专用", "避免传染"],
        "foods": [], "avoid_foods": ["辛辣食物"],
        "category": {"左氧氟沙星滴眼液": "喹诺酮类抗生素", "色甘酸钠滴眼液": "抗过敏药"},
    },
    "湿疹": {
        "description": "由多种内外因素引起的具有明显渗出倾向的皮肤炎症反应，瘙痒剧烈。",
        "symptoms": ["红斑", "丘疹", "瘙痒", "渗出"],
        "department": "皮肤科",
        "medications": ["卤米松乳膏", "氯雷他定"],
        "examinations": ["过敏原检测"],
        "body_parts": ["皮肤"],
        "precautions": ["避免热水烫洗", "保湿护肤", "穿宽松棉质衣物"],
        "foods": [], "avoid_foods": ["辛辣食物", "酒精", "已知过敏食物"],
        "category": {"卤米松乳膏": "外用糖皮质激素", "氯雷他定": "抗组胺药"},
    },
    "荨麻疹": {
        "description": "皮肤黏膜小血管扩张及渗透性增加的局限性水肿反应，风团时起时消。",
        "symptoms": ["风团", "瘙痒", "血管性水肿"],
        "department": "皮肤科",
        "medications": ["氯雷他定", "西替利嗪"],
        "examinations": ["过敏原检测", "血常规"],
        "body_parts": ["皮肤"],
        "precautions": ["寻找并去除病因", "记录发作诱因"],
        "foods": [], "avoid_foods": ["海鲜", "酒精"],
        "category": {"氯雷他定": "抗组胺药", "西替利嗪": "抗组胺药"},
    },
    "尿路感染": {
        "description": "病原体在尿路中生长繁殖，侵犯尿路黏膜或组织引起的感染性疾病。",
        "symptoms": ["尿频", "尿急", "尿痛", "发热"],
        "department": "肾内科",
        "medications": ["左氧氟沙星", "头孢克肟"],
        "examinations": ["尿常规", "尿培养"],
        "body_parts": ["膀胱", "尿道", "肾"],
        "precautions": ["多喝水勤排尿", "注意会阴清洁"],
        "foods": ["大量饮水", "蔓越莓"], "avoid_foods": ["辛辣食物"],
        "category": {"左氧氟沙星": "喹诺酮类抗生素", "头孢克肟": "头孢菌素类抗生素"},
    },
    "肾结石": {
        "description": "晶体物质在肾脏异常聚积形成的结石，腰痛血尿为典型表现。",
        "symptoms": ["腰痛", "血尿", "恶心", "排尿困难"],
        "department": "泌尿外科",
        "medications": ["坦索罗辛", "双氯芬酸钠"],
        "examinations": ["泌尿系B超", "尿常规", "CT"],
        "body_parts": ["肾", "输尿管", "膀胱"],
        "precautions": ["大量饮水", "根据结石成分调整饮食"],
        "foods": ["大量饮水", "柑橘类水果"], "avoid_foods": ["高草酸食物", "高嘌呤食物"],
        "category": {"坦索罗辛": "α受体阻滞剂", "双氯芬酸钠": "非甾体抗炎药"},
    },
    "前列腺炎": {
        "description": "前列腺受病原体感染或某些非感染因素刺激发生的炎症反应。",
        "symptoms": ["尿频", "尿急", "会阴部胀痛", "排尿不适"],
        "department": "泌尿外科",
        "medications": ["左氧氟沙星", "坦索罗辛"],
        "examinations": ["前列腺液检查", "前列腺B超"],
        "body_parts": ["前列腺", "尿道"],
        "precautions": ["避免久坐", "规律性生活", "戒烟酒", "温水坐浴"],
        "foods": [], "avoid_foods": ["酒精", "辛辣食物"],
        "category": {"左氧氟沙星": "喹诺酮类抗生素", "坦索罗辛": "α受体阻滞剂"},
    },
    "月经不调": {
        "description": "月经周期、经期、经量异常的一类妇科常见疾病。",
        "symptoms": ["周期紊乱", "经量异常", "痛经"],
        "department": "妇产科",
        "medications": ["黄体酮", "乌鸡白凤丸"],
        "examinations": ["性激素六项", "妇科B超"],
        "body_parts": ["子宫", "卵巢"],
        "precautions": ["规律作息", "注意保暖", "调节情绪"],
        "foods": ["红枣", "红糖姜茶"], "avoid_foods": ["生冷食物"],
        "category": {"黄体酮": "孕激素类", "乌鸡白凤丸": "中成药"},
    },
    "抑郁症": {
        "description": "以显著而持久的心境低落为主要特征的心理疾病，可伴自杀观念。",
        "symptoms": ["情绪低落", "兴趣减退", "睡眠障碍", "食欲改变", "乏力"],
        "department": "精神心理科",
        "medications": ["舍曲林", "氟西汀"],
        "examinations": ["心理测评量表", "甲状腺功能检查"],
        "body_parts": ["大脑"],
        "precautions": ["及时就医", "坚持治疗", "社会支持", "规律运动"],
        "foods": ["富含Omega-3食物"], "avoid_foods": ["酒精"],
        "category": {"舍曲林": "SSRI类抗抑郁药", "氟西汀": "SSRI类抗抑郁药"},
    },
    "焦虑症": {
        "description": "以过度、持续的焦虑和担忧为核心表现的精神心理疾病。",
        "symptoms": ["过度担忧", "心悸", "出汗", "坐立不安", "失眠"],
        "department": "精神心理科",
        "medications": ["帕罗西汀", "丁螺环酮"],
        "examinations": ["心理测评量表", "甲状腺功能检查"],
        "body_parts": ["大脑"],
        "precautions": ["放松训练", "规律运动", "减少咖啡因"],
        "foods": [], "avoid_foods": ["咖啡", "浓茶"],
        "category": {"帕罗西汀": "SSRI类抗抑郁药", "丁螺环酮": "抗焦虑药"},
    },
    "痔疮": {
        "description": "肛垫下移和静脉丛淤血曲张形成的肛肠疾病，便血和脱出为常见表现。",
        "symptoms": ["便血", "肛门肿物", "肛门瘙痒", "疼痛"],
        "department": "消化内科",
        "medications": ["马应龙麝香痔疮膏", "地奥司明"],
        "examinations": ["肛门指检", "肛门镜"],
        "body_parts": ["肛门", "直肠"],
        "precautions": ["避免久坐久蹲", "保持大便通畅", "温水坐浴"],
        "foods": ["高纤维食物", "大量饮水"], "avoid_foods": ["辛辣食物", "酒精"],
        "category": {"马应龙麝香痔疮膏": "外用中成药", "地奥司明": "静脉活性药物"},
    },
    "甲状腺功能亢进": {
        "description": "甲状腺腺体本身产生甲状腺激素过多引起的甲状腺毒症，简称甲亢。",
        "symptoms": ["心悸", "多汗", "体重下降", "手抖", "易怒"],
        "department": "内分泌科",
        "medications": ["甲巯咪唑", "普萘洛尔"],
        "examinations": ["甲状腺功能检查", "甲状腺B超"],
        "body_parts": ["甲状腺", "心脏"],
        "precautions": ["低碘饮食", "定期复查甲功", "避免劳累"],
        "foods": ["高热量高蛋白饮食"], "avoid_foods": ["海带", "紫菜", "含碘盐"],
        "category": {"甲巯咪唑": "抗甲状腺药", "普萘洛尔": "β受体阻滞剂"},
    },
    "骨质疏松": {
        "description": "骨量减少、骨微结构破坏，骨脆性增加易骨折的全身性骨病。",
        "symptoms": ["腰背痛", "身高变矮", "驼背", "易骨折"],
        "department": "骨科",
        "medications": ["阿仑膦酸钠", "钙尔奇D", "骨化三醇"],
        "examinations": ["骨密度检测", "血钙", "维生素D"],
        "body_parts": ["骨骼", "脊柱"],
        "precautions": ["补钙和维生素D", "适度负重运动", "防跌倒"],
        "foods": ["牛奶", "豆制品", "深绿色蔬菜"], "avoid_foods": ["过量咖啡", "碳酸饮料"],
        "category": {"阿仑膦酸钠": "双膦酸盐类", "钙尔奇D": "钙剂", "骨化三醇": "活性维生素D"},
    },
    "牙周炎": {
        "description": "发生在牙周支持组织的慢性炎症，是成人牙齿丧失的首位原因。",
        "symptoms": ["牙龈出血", "口臭", "牙齿松动", "牙龈退缩"],
        "department": "口腔科",
        "medications": ["甲硝唑", "阿莫西林"],
        "examinations": ["牙周探诊", "X线片"],
        "body_parts": ["牙齿", "牙龈", "牙槽骨"],
        "precautions": ["正确刷牙", "使用牙线", "定期洁牙"],
        "foods": ["富含维生素C食物"], "avoid_foods": ["过甜食物"],
        "category": {"甲硝唑": "抗厌氧菌药", "阿莫西林": "青霉素类抗生素"},
    },
    "带状疱疹": {
        "description": "水痘-带状疱疹病毒再激活引起的感染性皮肤病，沿神经分布的簇集水疱伴神经痛。",
        "symptoms": ["簇集水疱", "神经痛", "皮肤灼热感"],
        "department": "皮肤科",
        "medications": ["阿昔洛韦", "加巴喷丁", "甲钴胺"],
        "examinations": ["临床查体", "水痘-带状疱疹病毒抗体检测"],
        "body_parts": ["皮肤", "神经"],
        "precautions": ["早期抗病毒治疗", "避免接触孕妇和儿童", "注意休息"],
        "foods": ["高蛋白饮食"], "avoid_foods": ["辛辣食物"],
        "category": {"阿昔洛韦": "抗病毒药", "加巴喷丁": "神经痛治疗药", "甲钴胺": "神经营养药"},
    },
    "冠心病_心绞痛": {
        "description": "冠状动脉供血不足，心肌急剧的暂时性缺血缺氧引起的发作性胸痛综合征。",
        "symptoms": ["胸痛", "胸闷", "压榨感", "放射痛"],
        "department": "心血管内科",
        "medications": ["硝酸甘油", "单硝酸异山梨酯"],
        "examinations": ["心电图", "运动负荷试验", "冠状动脉造影"],
        "body_parts": ["心脏", "冠状动脉"],
        "precautions": ["随身携带硝酸甘油", "避免情绪激动", "避免饱餐后运动"],
        "foods": ["低脂饮食"], "avoid_foods": ["高脂饮食", "暴饮暴食"],
        "category": {"硝酸甘油": "硝酸酯类", "单硝酸异山梨酯": "硝酸酯类"},
    },
    "支气管扩张": {
        "description": "支气管壁结构破坏导致的异常扩张，慢性咳嗽、大量脓痰为特征。",
        "symptoms": ["慢性咳嗽", "大量脓痰", "反复咯血"],
        "department": "呼吸内科",
        "medications": ["氨溴索", "头孢他啶"],
        "examinations": ["胸部高分辨CT", "痰培养"],
        "body_parts": ["支气管", "肺"],
        "precautions": ["体位引流排痰", "预防感染", "戒烟"],
        "foods": ["高蛋白高维生素饮食"], "avoid_foods": ["辛辣刺激食物"],
        "category": {"氨溴索": "祛痰药", "头孢他啶": "头孢菌素类抗生素"},
    },
    "胆囊炎": {
        "description": "胆囊发生的急慢性化学性或细菌性炎症，多与胆结石并存。",
        "symptoms": ["右上腹痛", "恶心", "呕吐", "发热", "墨菲征阳性"],
        "department": "消化内科",
        "medications": ["头孢哌酮", "熊去氧胆酸"],
        "examinations": ["腹部B超", "血常规", "肝功能检查"],
        "body_parts": ["胆囊", "胆道"],
        "precautions": ["低脂饮食", "规律进餐", "及时处理胆结石"],
        "foods": ["低脂饮食", "蔬菜"], "avoid_foods": ["油腻食物", "暴饮暴食"],
        "category": {"头孢哌酮": "头孢菌素类抗生素", "熊去氧胆酸": "利胆药"},
    },
}


def build_seed() -> None:
    diseases: list[dict] = []
    entities: list[dict] = []
    relations: list[dict] = []

    def add_entity(etype: str, name: str = "", **props) -> None:
        entity = {"type": etype, "name": name, **props}
        if entity not in entities:
            entities.append(entity)

    for name, info in SEED.items():
        display_name = name.replace("_", "")
        diseases.append({
            "name": display_name,
            "description": info["description"],
            "symptoms": info["symptoms"],
            "treatment": "",
            "department": info["department"],
            "medications": info["medications"],
            "precautions": "；".join(info["precautions"]),
            "url": "",
        })
        add_entity("Disease", display_name, description=info["description"], icd_code=None)

        for symptom in info["symptoms"]:
            add_entity("Symptom", symptom, severity="")
            relations.append({
                "type": "HAS_SYMPTOM",
                "data": {"disease_name": display_name, "symptom_name": symptom},
            })
            # R6 单向关系：症状→疾病不建 MAY_INDICATE 反向边，反查用 Cypher 反向遍历
        add_entity("Department", info["department"], floor="")
        relations.append({
            "type": "BELONGS_TO_DEPARTMENT",
            "data": {"disease_name": display_name, "department_name": info["department"]},
        })
        for med in info["medications"]:
            category = info.get("category", {}).get(med, "")
            add_entity("Medication", med, category=category, side_effects=[], contraindications=[])
            relations.append({
                "type": "TREATED_BY_MEDICATION",
                "data": {"disease_name": display_name, "drug_name": med},
            })
        for exam in info["examinations"]:
            add_entity("Examination", exam)
            relations.append({
                "type": "NEEDS_EXAMINATION",
                "data": {"disease_name": display_name, "examination_name": exam},
            })
        for part in info["body_parts"]:
            add_entity("BodyPart", part)
            relations.append({
                "type": "AFFECTS_BODY_PART",
                "data": {"disease_name": display_name, "body_part_name": part},
            })
        for precaution in info["precautions"]:
            if precaution:
                add_entity("Precaution", content=precaution)
                relations.append({
                    "type": "HAS_PRECAUTION",
                    "data": {"disease_name": display_name, "precaution_content": precaution},
                })
        for food in info.get("foods", []):
            add_entity("Food", food)
            relations.append({
                "type": "RECOMMENDS_FOOD",
                "data": {"disease_name": display_name, "food_name": food, "recommendation_type": "推荐"},
            })
        for food in info.get("avoid_foods", []):
            add_entity("Food", food)
            relations.append({
                "type": "RECOMMENDS_FOOD",
                "data": {"disease_name": display_name, "food_name": food, "recommendation_type": "禁忌"},
            })

    # 药物相互作用（策划常识对：阿司匹林-华法林等经典组合只放演示用安全对）
    interactions = [("阿司匹林", "华法林"), ("阿司匹林", "布洛芬")]
    for a, b in interactions:
        if any(e["name"] == a for e in entities) and any(e["name"] == b for e in entities):
            relations.append({
                "type": "INTERACTS_WITH",
                "data": {"drug_name": a, "interacting_drug_name": b},
            })
    add_entity("Medication", "华法林", category="抗凝药", side_effects=[], contraindications=[])

    # 副作用关系（经典常识）
    side_effects = [("阿司匹林", "胃部不适"), ("布洛芬", "胃部不适"), ("氨氯地平", "踝部水肿")]
    for drug, se in side_effects:
        add_entity("Symptom", se, severity="轻度")
        relations.append({
            "type": "HAS_SIDE_EFFECT",
            "data": {"drug_name": drug, "symptom_name": se},
        })

    # 去重实体
    unique: list[dict] = []
    seen: set = set()
    for entity in entities:
        key = (entity["type"], entity.get("name") or entity.get("content", ""))
        if key not in seen:
            seen.add(key)
            unique.append(entity)

    write_json(PROCESSED_DISEASES_JSON, {"diseases": diseases})
    write_json(KG_ENTITIES_FILE, {"entities": unique})
    write_json(KG_RELATIONS_FILE, {"relations": relations})

    print(f"✅ 种子数据生成完成（{len(diseases)} 疾病 / {len(unique)} 实体 / {len(relations)} 关系）")
    print(f"   {PROCESSED_DISEASES_JSON}")
    print(f"   {KG_ENTITIES_FILE}")
    print(f"   {KG_RELATIONS_FILE}")
    print("下一步: python src/knowledge_graph/graph_builder.py build && python scripts/build_index.py")


if __name__ == "__main__":
    build_seed()
