# pip install pymupdf

import fitz  # PyMuPDF
# 获取全书的所有图片，现在已经将全书的所有图片保存在massive/pic/allpic目录中了

# 打开 PDF 文件
pdf_document = ""
pdf = fitz.open(pdf_document)

# 遍历每一页
for page_num in range(len(pdf)):
    page = pdf.load_page(page_num)
    image_list = page.get_images(full=True)
    
    # 打印页面中的图片数
    print(f"Page {page_num + 1} contains {len(image_list)} image(s).")
    
    for image_index, img in enumerate(page.get_images(full=True)):
        xref = img[0]
        base_image = pdf.extract_image(xref)
        image_bytes = base_image["image"]
        image_ext = base_image["ext"]
        image_name = f"page{page_num + 1}_img{image_index + 1}.{image_ext}"
        
        # 将图片保存为文件
        with open(image_name, "wb") as img_file:
            img_file.write(image_bytes)
            
        print(f"Saved {image_name}")
