# OmniDocBench

[English](./README.md) | [简体中文](./README_ZH.md)

**OmniDocBench** is an evaluation dataset for diverse document parsing in real-world scenarios, with the following characteristics:
- **Diverse Document Types**: The evaluation set contains 1355 PDF pages, covering 9 document types, 4 layout types and 3 language types. It has broad coverage including academic papers, financial reports, newspapers, textbooks, handwritten notes, etc.
- **Rich Annotations**: Contains location information for 15 block-level (text paragraphs, titles, tables, etc., over 20k in total) and 4 span-level (text lines, inline formulas, superscripts/subscripts, etc., over 80k in total) document elements, as well as recognition results for each element region (text annotations, LaTeX formula annotations, tables with both LaTeX and HTML annotations). OmniDocBench also provides reading order annotations for document components. Additionally, it includes various attribute labels at page and block levels, with 5 page attribute labels, 3 text attribute labels and 6 table attribute labels.
- **High Annotation Quality**: Through manual screening, intelligent annotation, manual annotation, full expert quality inspection and large model quality inspection, the data quality is relatively high.
- **Evaluation Code Suite**: Designed with end-to-end evaluation and single module evaluation code to ensure fairness and accuracy of evaluation. The evaluation code suite can be found at [OmniDocBench](https://github.com/opendatalab/OmniDocBench).

## Updates

- [2025/09/25] Major Update: updated from v1.0 to v1.5:
    - Images of newspaper and notes have been increased to 200 DPI.
    - To balance the number of pages in Chinese and English and increase the number of pages containing formulas, 374 new pages have been added, including 25 in Chinese and 349 in English. These pages include books, PPTs, color illustrated textbooks, test papers, magazines, and newspapers. The number of inline formulas has increased from 353 to 1050.
    - Language attributes have been added to formulas, increasing the number of Chinese formulas to 68 and English formulas to 982.
    - Fixed typos in some text and table annotations in v1.0.
- [2024/12/25] Added PDF format of the evaluation set for models that require PDFs as input for evaluation. Added original PDF slices with metadata.
- [2024/12/10] Fixed height and width fields for some samples. This fix only affects page-level height and width fields and does not impact the correctness of other annotations
- [2024/12/04] Released OmniDocBench evaluation dataset

## Dataset Introduction

The evaluation set contains 1355 PDF pages, covering 9 document types, 4 layout types and 3 language types. OmniDocBench has rich annotations, including 15 block-level annotations (text paragraphs, titles, tables, etc.) and 4 span-level annotations (text lines, inline formulas, superscripts/subscripts, etc.). All text-related annotation boxes contain text recognition annotations, formulas contain LaTeX annotations, and tables contain both LaTeX and HTML annotations. OmniDocBench also provides reading order annotations for document components. Additionally, it includes various attribute labels at page and block levels, with 5 page attribute labels, 3 text attribute labels and 6 table attribute labels.

![](data_diversity.png)

## Usage

You can use our [evaluation method](https://github.com/opendatalab/OmniDocBench) to conduct evaluations across several dimensions:

- End-to-end evaluation: Includes both end2end and md2md evaluation methods
- Layout detection
- Table recognition
- Formula recognition
- Text OCR

The evaluation dataset files include:

- [OmniDocBench.json](OmniDocBench.json) is the annotation file for the evaluation dataset, stored in JSON format. It supports the end2end evaluation method. The structure and fields are explained below.
- [images](./images/) are the corresponding evaluation dataset images, for models that require images as input.
- [image_to_pdf.py](https://github.com/opendatalab/OmniDocBench/blob/main/tools/image_to_pdf.py) is the script to covert images to PDFs for those models who take only PDFs as input.


<details>
  <summary>Dataset Format</summary>

The dataset format is JSON, with the following structure and field explanations:

```json
[{
    "layout_dets": [    // List of page elements
        {
            "category_type": "text_block",  // Category name
            "poly": [
                136.0, // Position information, coordinates for top-left, top-right, bottom-right, bottom-left corners (x,y)
                781.0,
                340.0,
                781.0,
                340.0,
                806.0,
                136.0,
                806.0
            ],
            "ignore": false,        // Whether to ignore during evaluation
            "order": 0,             // Reading order
            "anno_id": 0,           // Special annotation ID, unique for each layout box
            "text": "xxx",          // Optional field, Text OCR results are written here
            "latex": "$xxx$",       // Optional field, LaTeX for formulas and tables is written here
            "html": "xxx",          // Optional field, HTML for tables is written here
            "attribute" {"xxx": "xxx"},         // Classification attributes for layout, detailed below
            "line_with_spans:": [   // Span level annotation boxes
                {
                    "category_type": "text_span",
                    "poly": [...],
                    "ignore": false,
                    "text": "xxx",   
                    "latex": "$xxx$",
                 },
                 ...
            ],
            "merge_list": [    // Only present in annotation boxes with merge relationships, merge logic depends on whether single line break separated paragraphs exist, like list types
                {
                    "category_type": "text_block", 
                    "poly": [...],
                    ...   // Same fields as block level annotations
                    "line_with_spans": [...]
                    ...
                 },
                 ...
            ]
        ...
    ],
    "page_info": {         
        "page_no": 0,            // Page number
        "height": 1684,          // Page height
        "width": 1200,           // Page width
        "image_path": "xx/xx/",  // Annotated page filename
        "page_attribute": {"xxx": "xxx"}     // Page attribute labels
    },
    "extra": {
        "relation": [ // Related annotations
            {  
                "source_anno_id": 1,
                "target_anno_id": 2, 
                "relation": "parent_son"  // Relationship label between figure/table and their corresponding caption/footnote categories
            },
            {  
                "source_anno_id": 5,
                "target_anno_id": 6,
                "relation_type": "truncated"  // Paragraph truncation relationship label due to layout reasons, will be concatenated and evaluated as one paragraph during evaluation
            },
        ]
    }
},
...
]
```

</details>

<details>
  <summary>Evaluation Categories</summary>

Evaluation categories include:

```
# Block level annotation boxes
'title'               # Title
'text_block'          # Paragraph level plain text
'figure',             # Figure type
'figure_caption',     # Figure description/title
'figure_footnote',    # Figure notes
'table',              # Table body
'table_caption',      # Table description/title
'table_footnote',     # Table notes
'equation_isolated',  # Display formula
'equation_caption',   # Formula number
'header'              # Header
'footer'              # Footer
'page_number'         # Page number
'page_footnote'       # Page notes
'abandon',            # Other discarded content (e.g. irrelevant information in middle of page)
'code_txt',           # Code block
'code_txt_caption',   # Code block description
'reference',          # References

# Span level annotation boxes
'text_span'           # Span level plain text
'equation_ignore',    # Formula to be ignored
'equation_inline',    # Inline formula
'footnote_mark',      # Document superscripts/subscripts
```

</details>

<details>
  <summary>Attribute Labels</summary>

Page classification attributes include:

```
'data_source': #PDF type classification
    academic_literature  # Academic literature
    PPT2PDF # PPT to PDF
    book # Black and white books and textbooks
    colorful_textbook # Colorful textbooks with images
    exam_paper # Exam papers
    note # Handwritten notes
    magazine # Magazines
    research_report # Research reports and financial reports
    newspaper # Newspapers

'language': #Language type
    en # English
    simplified_chinese # Simplified Chinese
    en_ch_mixed # English-Chinese mixed

'layout': #Page layout type
    single_column # Single column
    double_column # Double column
    three_column # Three column
    1andmore_column # One mixed with multiple columns, common in literature
    other_layout # Other layouts

'watermark': # Whether contains watermark
    true  
    false

'fuzzy_scan': # Whether blurry scanned
    true  
    false

'colorful_backgroud': # Whether contains colorful background, content to be recognized has more than two background colors
    true  
    false
```

Block level attribute - Table related attributes:

```
'table_layout': # Table orientation
    vertical # Vertical table
    horizontal # Horizontal table

'with_span': # Merged cells
    False
    True

'line': # Table borders
    full_line # Full borders
    less_line # Partial borders
    fewer_line # Three-line borders
    wireless_line # No borders

'language': # Table language
    table_en # English table
    table_simplified_chinese # Simplified Chinese table
    table_en_ch_mixed # English-Chinese mixed table

'include_equation': # Whether table contains formulas
    False
    True

'include_backgroud': # Whether table contains background color
    False
    True

'table_vertical' # Whether table is rotated 90 or 270 degrees
    False
    True
```

Block level attribute - Text paragraph related attributes:

```
'text_language': # Text language
    text_en  # English
    text_simplified_chinese # Simplified Chinese
    text_en_ch_mixed  # English-Chinese mixed

'text_background':  # Text background color
    white # Default value, white background
    single_colored # Single background color other than white
    multi_colored  # Multiple background colors

'text_rotate': # Text rotation classification within paragraphs
    normal # Default value, horizontal text, no rotation
    rotate90  # Rotation angle, 90 degrees clockwise
    rotate180 # 180 degrees clockwise
    rotate270 # 270 degrees clockwise
    horizontal # Text is normal but layout is vertical
```

Block level attribute - Formula related attributes:

```
'formula_type': # Formula type
    print  # Print
    handwriting # Handwriting

'equation_language' # Formula language
    equation_en  # English
    equation_ch # Chinese
```

</details>


## Data Display
![](show_pdf_types_1.png)
![](show_pdf_types_2.png)

## Acknowledgement

- Thank [Abaka AI](https://abaka.ai) for supporting the dataset annotation.

## Copyright Statement

The PDFs are collected from public online channels and community user contributions. Content that is not allowed for distribution has been removed. The dataset is for research purposes only and not for commercial use. If there are any copyright concerns, please contact OpenDataLab@pjlab.org.cn.

## Citation

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

## Links

- Paper: https://huggingface.co/papers/2412.07626
- GitHub: https://github.com/opendatalab/OmniDocBench