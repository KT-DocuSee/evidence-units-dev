# OmniDocBench

[English](./README.md) | 简体中文

**OmniDocBench**是一个针对真实场景下多样性文档解析评测集，具有以下特点：
- **文档类型多样**：该评测集涉及1355个PDF页面，涵盖9种文档类型、4种排版类型和3种语言类型。覆盖面广，包含学术文献、财报、报纸、教材、手写笔记等；
- **标注信息丰富**：包含15个block级别（文本段落、标题、表格等，总量超过20k）和4个Span级别（文本行、行内公式、角标等，总量超过80k）的文档元素的**定位信息**，以及每个元素区域的**识别结果**（文本Text标注，公式LaTeX标注，表格包含LaTeX和HTML两种类型的标注）。OmniDocBench还提供了各个文档组件的**阅读顺序**的标注。除此之外，在页面和block级别还包含多种属性标签，标注了5种**页面属性标签**、3种**文本属性标签**和6种**表格属性标签**。
- **标注质量高**：经过人工筛选，智能标注，人工标注及全量专家质检和大模型质检，数据质量较高。
- **配套评测代码**：设计端到端评测及单模块评测代码，保证评测的公平性及准确性。配套的评测代码请访问[OmniDocBench](https://github.com/opendatalab/OmniDocBench)。

## 更新

- [2025/09/25] 重大更新：从版本1.0更新到1.5:
    - 报纸和笔记类型的图片提升到200DPI；
    - 为了平衡中英文页面数量，并提升包含公式页面的数量，新增了374个页面，其中中文增加了25个页面，英文增加了349个页面，包括书籍类、PPT类、彩色图文教材类、试卷、杂志和报纸类，行间公式数量从353个增加到1050个；
    - 公式增加了语种属性，行间的中文公式数量增至68个，英文公式数量增至982个；
    - 修复1.0版本的OCR和table的部分拼写错误；
- [2024/12/25] 新增了评测集的PDF格式，供需要PDF作为输入的模型进行评测。新增了包含元信息的原始PDF切片。
- [2024/12/10] 修正了部分样本height和width字段，该修正仅涉及页面级别的height和width字段，不影响其他标注的正确性。
- [2024/12/04] OmniDocBench评测集发布。

## 评测集介绍

该评测集涉及1355个PDF页面，涵盖9种文档类型、4种排版类型和3种语言类型。OmniDocBench具有丰富的标注，包含15个block级别的标注（文本段落、标题、表格等）和4个Span级别的标注（文本行、行内公式、角标等）。所有文本相关的标注框上都包含文本识别的标注，公式包含LaTeX标注，表格包含LaTeX和HTML两种类型的标注。OmniDocBench还提供了各个文档组件的阅读顺序的标注。除此之外，在页面和block级别还包含多种属性标签，标注了5种页面属性标签、3种文本属性标签和6种表格属性标签。

![](data_diversity.png)


## 使用

评测可以使用我们提供的[评测脚本](https://github.com/opendatalab/OmniDocBench), 可进行以下几个维度的评测：

- 端到端评测：包括end2end和md2md两种评测方式
- Layout检测
- 表格识别
- 公式识别
- 文本OCR

评测集的文件包括：

- [OmniDocBench.json](OmniDocBench.json) 是评测集的标注文件，以JSON格式存储，支持end2end的评测方式，其结构和字段在后文有解释。
- [images](./images/) 是对应的评测集图像，供需要图片作为输入的模型进行评测。
- [image_to_pdf.py](https://github.com/opendatalab/OmniDocBench/blob/main/tools/image_to_pdf.py) 是将图片转换成PDF的脚本，主要是为了部分模型需要以PDF作为输入而提供的。

<details>
  <summary>评测集的数据格式</summary>

评测集的数据格式为JSON，其结构和各个字段的解释如下：

```json
[{
    "layout_dets": [    // 页面元素列表
        {
            "category_type": "text_block",  // 类别名称
            "poly": [
                136.0, // 位置信息，分别是左上角、右上角、右下角、左下角的x,y坐标
                781.0,
                340.0,
                781.0,
                340.0,
                806.0,
                136.0,
                806.0
            ],
            "ignore": false,        // 是否在评测的时候不考虑
            "order": 0,             // 阅读顺序
            "anno_id": 0,           // 特殊的标注ID，每个layout框唯一
            "text": "xxx",          // 可选字段，OCR结果会写在这里
            "latex": "$xxx$",       // 可选字段，formula和table的LaTeX会写在这里
            "html": "xxx",          // 可选字段，table的HTML会写在这里
            "attribute" {"xxx": "xxx"},         // layout的分类属性，后文会详细展示
            "line_with_spans:": [   // span level的标注框
                {
                    "category_type": "text_span",
                    "poly": [...],
                    "ignore": false,
                    "text": "xxx",   
                    "latex": "$xxx$",
                 },
                 ...
            ],
            "merge_list": [    // 只有包含merge关系的标注框内有这个字段，是否包含merge逻辑取决于是否包含单换行分割小段落，比如列表类型
                {
                    "category_type": "text_block", 
                    "poly": [...],
                    ...   // 跟block级别标注的字段一致
                    "line_with_spans": [...]
                    ...
                 },
                 ...
            ]
        ...
    ],
    "page_info": {         
        "page_no": 0,            // 页码
        "height": 1684,          // 页面的宽
        "width": 1200,           // 页面的高
        "image_path": "xx/xx/",  // 标注的页面文件名称
        "page_attribute": {"xxx": "xxx"}     // 页面的属性标签
    },
    "extra": {
        "relation": [ // 具有相关关系的标注
            {  
                "source_anno_id": 1,
                "target_anno_id": 2, 
                "relation": "parent_son"  // figure/table与其对应的caption/footnote类别的关系标签
            },
            {  
                "source_anno_id": 5,
                "target_anno_id": 6,
                "relation_type": "truncated"  // 段落因为排版原因被截断，会标注一个截断关系标签，后续评测的时候会拼接后再作为一整个段落进行评测
            },
        ]
    }
},
...
]
```

</details>

<details>
  <summary>验证集类别</summary>

验证集类别包括：

```
# Block级别标注框
'title'               # 标题
'text_block'          # 段落级别纯文本
'figure',             # 图片类
'figure_caption',     # 图片说明、标题
'figure_footnote',    # 图片注释
'table',              # 表格主体
'table_caption',      # 表格说明和标题
'table_footnote',     # 表格的注释
'equation_isolated',  # 行间公式
'equation_caption',   # 公式序号
'header'              # 页眉
'footer'              # 页脚  
'page_number'         # 页码
'page_footnote'       # 页面注释
'abandon',            # 其他的舍弃类（比如页面中间的一些无关信息）
'code_txt',           # 代码块
'code_txt_caption',   # 代码块说明
'reference',          # 参考文献类

# Span级别标注框
'text_span'           # span级别的纯文本
'equation_ignore',    # 需要忽略的公式类
'equation_inline',    # 行内公式类
'footnote_mark',      #文章的上下角标
```

</details>

<details>
  <summary>验证集属性标签</summary>

页面分类属性包括：
```
'data_source': #PDF类型分类
    academic_literature  # 学术文献
    PPT2PDF # PPT转PDF
    book # 黑白的图书和教材
    colorful_textbook # 彩色图文教材
    exam_paper # 试卷
    note # 手写笔记
    magazine # 杂志
    research_report # 研报、财报
    newspaper # 报纸

'language':#语种
    en # 英文
    simplified_chinese # 简体中文
    en_ch_mixed # 中英混合

'layout': #页面布局类型
    single_column # 单栏
    double_column # 双栏
    three_column # 三栏
    1andmore_column # 一混多，常见于文献
    other_layout # 其他

'watermark'： # 是否包含水印
    true  
    false

'fuzzy_scan': # 是否模糊扫描
    true  
    false

'colorful_backgroud': # 是否包含彩色背景，需要参与识别的内容的底色包含两个以上
    true  
    false
```

标注框级别属性-表格相关属性:

```
'table_layout': # 表格的方向
    vertical #竖版表格
    horizontal #横版表格

'with_span': # 合并单元格
    False
    True

'line':# 表格的线框
    full_line # 全线框
    less_line # 漏线框
    fewer_line # 三线框 
    wireless_line # 无线框

'language': #表格的语种
    table_en  # 英文表格
    table_simplified_chinese  #中文简体表格
    table_en_ch_mixed  #中英混合表格

'include_equation': # 表格是否包含公式
    False
    True

'include_backgroud': # 表格是否包含底色
    False
    True

'table_vertical' # 表格是否旋转90度或270度
    False
    True
```

标注框级别属性-文本段落相关属性: 
```
'text_language': # 文本的段落内语种
    text_en  # 英文
    text_simplified_chinese #简体中文
    text_en_ch_mixed  #中英混合

'text_background':  #文本的背景色
    white # 默认值，白色背景
    single_colored # 除白色外的单背景色
    multi_colored  # 混合背景色

'text_rotate': # 文本的段落内文字旋转分类
    normal # 默认值，横向文本，没有旋转
    rotate90  # 旋转角度，顺时针旋转90度
    rotate180 # 顺时针旋转180度
    rotate270 # 顺时针旋转270度
    horizontal # 文字是正常的，排版是竖型文本
```

标注框级别属性-公式相关属性: 
```
'formula_type': #公式类型
    print  # 打印体
    handwriting # 手写体
        
'equation_language' # Formula language
    equation_en  # English
    equation_ch # Chinese
```

</details>

## 数据展示
![](show_pdf_types_1.png)
![](show_pdf_types_2.png)

## Acknowledgement

- 感谢[Abaka AI](https://abaka.ai)支持数据集标注。

## 版权声明
  
PDF来源从网络公开渠道收集以及社群用户贡献，已剔除了不允许分发的内容，只用作科研，不作为商业用途。若有侵权请联系OpenDataLab@pjlab.org.cn。

## 引用

```bibtex
@misc{ouyang2024omnidocbenchbenchmarkingdiversepdf,
      title={OmniDocBench: Benchmarking Diverse PDF Document Parsing with Comprehensive Annotations}, 
      author={Linke Ouyang and Yuan Qu and Hongbin Zhou and Jiawei Zhu and Rui Zhang and Qunshu Lin and Bin Wang and Zhiyuan Zhao and Man Jiang and Xiaomeng Zhao and Jin Shi and Fan Wu and Pei Chu and Minghao Liu and Zhenxiang Li and Chao Xu and Bo Zhang and Botian Shi and Zhongying Tu and Conghui He},
      year={2024},
      eprint={2412.07626},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2412.07626}, 
}
```

## 相关链接
- 论文: https://huggingface.co/papers/2412.07626
- GitHub: https://github.com/opendatalab/OmniDocBench