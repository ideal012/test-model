# Folder Project (ตัวอย่างโครงสร้างของโปรเจค แต่นศ.สามารถออกแบบรูปแบบอื่นๆได้ เช่น MVC/Solid Desgin Pattern หรืออื่น ๆ)demoWebApp/
    app.py             (ตัวโครงหลักเว็บที่ทำด้วย Streamlit เพื่อรับ input แล้วแสดงผล)
    model.py         (ตัวโครง Class model ไม่มีการโหลด weight)
    inference.py   (ตัวกลางที่ช่วยในการสั่งโมเดล)
    utils.py            (ตัวที่ช่วยเก็บฟังก์ชั่นช่วยเหลือต่าง ๆ เช่นพวก pre-procress / post-procress)
    /weights/
        - model.pth or model.onnx
