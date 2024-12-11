# BiTNet : AI for Ultrasound Image Classification

Introduction :

ในปัจจุบันในอาเซียน ชาวบ้านมักกินปลาน้ำน้ำจืดแต่ไม่ปรุงสุก ทำให้เกิดโรคพยาธิใบไม้ในตับ ส่งผลให้เป็นมะเร็งท่อน้ำดี ซึ่งโดยส่วนมากอัตราการรอดชีวิตอยู่ที่ 2-5 ปี แต่เราสามารถตรวจพบได้ไวกว่านั้น ด้วย 
Early-stage CCA เราสามารถจัดการได้โดย Ultrasound แต่ !! มันใช้ผู้ที่เชี่ยวชาญมาก ในแถบทั่วแม่น้ำโขงคือขาดแคลนผู้เชี่ยวชาญมาก ทำให้การตรวจค่อนข้างยากลำบาก

เราจึงใช้ Image classification ด้วย EfficientNet คัดกรองรูปภาพ Ultrasound image แทนมนุษย์

## 1. Datasets การ

รูปมันขึ้นอยู่กับคนถ่าย ฉะนั้นเป็น Challenge ของ Datasets นี้ (CT MRI มันได้ภาพเดียวกันเลย Fix ภาพ)

![image](https://github.com/user-attachments/assets/60340ce7-afd2-47c8-85ce-1b799e64eb81)

สังเกตว่าเขามีมุมที่ถ่ายแกต่างกันอย่างชัดเจน
โดยอยู่ภาคอีสาน ภาคเหนือ มีพฤติกรรมการกินสุกๆดิบๆ มีการติดพยาธิใบไม้ในตับ และอายุสูง เราก็จะคัดกรอง 6 เดือนทีนึง โดยที่มีแพทย์นึงยังไม่ได้เฉพาะทาง

![image](https://github.com/user-attachments/assets/05c48a9a-b492-4ca7-9452-82d5db500bee)

ถ้าเราเจอคนติดก็จะสแกน CT MRI ต่อไป ซึ่งมันเป็นงานหนักสำหรับนักรังสีแพทย์

Tele-mediologists พอเราได้ฐานข้อมูลมาแล้ว เราก็ใช้รังสีแพทย์เชี่ยวชาญในการจำแนก โดยจำแนก 15 Class แรก โดยที่มีความผิดปกติก็จะแยกเป็น 14 Class
เราก็จะมี Data 6569 ภาพ ซึ่งบางอันก็มีน้อย มีคนเป็นน้อยทำให้ Imbalanced Datasets

มุมที่แพทย์ใช้ในการถ่ายแบกด้วยตามการแยกตามรังสีแพทย์

![image](https://github.com/user-attachments/assets/6c288a22-9717-40a1-b365-cb795202d915)

เราจำเป็นต้องสร้าง Meta Data ว่ารูปภาพที่ 1 เก็บอยู่ที่ไหน ตำแหน่งอะไร Class อะไร เคสที่เท่าไหร่
เราจะแบ่งเป็น 10 Fold แล้วก็มีการกระจายตำแหน่งเท่ากันด้วย อีกทั้งยังบอก Test Validation แล้วก็ Train

![image](https://github.com/user-attachments/assets/d6090e70-afbe-4dec-902e-3b83318b7e8c)

6569 ภาพจาก 819 เคส เราสามารถลดได้เป็น 5 มุม (นักรังสีแพทย์)

ต่อมาเราจะ Computer Vision Model เราก็จะลบ Remove BG Information เช่น พวกชื่อคนไข้ ข้อมูลคนไข้ ขั้นตอนแรกเลย เราจะต้องแยกสัญญาณอัลตร้าซาวด์ก็จะ Crop ออกมาก่อน

![image](https://github.com/user-attachments/assets/a844feb3-44a4-4f96-b17e-864a25ae7d98)

EfficientNet และ RandomForest

* Data Augmentation - ในการเพิ่มจำนวนข้อมูล ในเชิงของ Computer Vision ควรจะเรียนรู้ในความหลากหลาย เช่น
    1. Horizontal Shift - มีโอกาสที่จะเลื่อนในแนวนอนซ้ายขวาอย่างไรบ้าง
    2. Vertical Shift - เลื่อนขึ้นเลื่อนลง
    3. Rotation 30 องศา - ได้นิดหน่อย
    4. Bright
    5. Shear
    6. Zoom
    7. No Flip - เพราะว่าภาพมันไม่ Flip คว่ำลง มันเรียนรู้ไม่ได้ Flip แนวนอนก็ไม่ใช้ เพราะถ้าอวัยวะย้ายที่ มันจะไม่เวิร์ค

## 2. Model 
![image](https://github.com/user-attachments/assets/b8c228ac-68eb-47bd-9a36-e52e41897e2b)

EfficientNetB0 มันทำมาจาก Google ก็เรียกใช้ใน Tensorflow ได้เลย ซึ่งมีหลายขนาด มันมี Input Image ที่มันใหญ่ยิ่งใหญ่ยิ่งดี ใช้ขนาด 456 x 456 pixels เราเลยเรียกใช้ EfficientNet

เราใช้ RandomForest เพื่อทำนายความผิดปกติ

![image](https://github.com/user-attachments/assets/d094bf51-9926-4aaa-9ceb-58b613c1f424)



