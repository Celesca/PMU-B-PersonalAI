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

* Auto Pre-screening -> เอาให้คนดูภาพน้อยที่สุด เพื่อจะได้รวดเร็ว แล้วก็ AI แนะนำว่ายังไงบ้าง เวลาเราสร้าง Computer Vision เราต้อง Fine-tune โมเดลให้เหมาะกับวิธีนั้นๆ

![image](https://github.com/user-attachments/assets/14d40fd3-ce21-4bbb-9876-d5468ce09f85)

เรามั่นใจว่ามันเป็น Abnormal อันไหนไม่มั่นใจว่า 100% มันจะอยู่ลำดับสูงๆ เพื่อให้อาจารย์ผู้เชี่ยวชาญดูภาพนี้ก่อน ถ้ามันไม่จำเป็นก็ให้ Priority ต่ำๆก็ไม่ต้องดูตอนว่างๆ
สรุปคือเราส่ง Abnormal หา Radiologist (100% confidence normal or Otherwise)

![image](https://github.com/user-attachments/assets/d1e49e88-0d74-421f-b8be-c5e2f36f9d19)

มองจากเวิร์คโหลด ลดงานได้ 35%

หน้าตาเว็บที่บอก Explainable AI + Predict 15 Classes 

![image](https://github.com/user-attachments/assets/128b6c55-cb9b-4388-9997-fd8db75073ef)

GradCam - Debug Layer ว่า Neuron Node นั้นทำงานเป็นพิเศษ มันจะอยู่ช่วงสีสว่าง มีความ Active มาก เขาทำให้มันสามารถ Explainable AI เข้าไป Assumption
![image](https://github.com/user-attachments/assets/bdf8d569-51b1-49cb-9ba1-4d5c33203c71)

การใช้งาน โดยทำนาย 150 ภาพ แล้วก็ Washout ว่าไม่มีเครื่องมือช่วยจะเป็นยังไง ฉะนั้นเขาทดสอบเครื่องมือว่าคุณหมอเขาจะทำยังไง โดยมีความเชี่ยวชาญต่างกัน

![image](https://github.com/user-attachments/assets/66e629a7-8303-4827-b221-c763189fb396)

แพทย์ทั่วไป ไม่ได้เรียนเฉพาะทาง / เรียนเฉพาะทางรังสีแพทย์ / จบรังสีแพทย์ อาจารย์หมอ Non-hepatobiliary แต่ไม่ได้ชำนาญการวิเคราะห์ช่องท้องส่วนบน / Hepatobiliary radiologists คุณหมอที่ชำนาญ

EfficientNet -> มันแม่นยำมาก / เมื่อเทียบกับข้อมูลเท็จสิ่งที่เกิดขึ้นคือ

![image](https://github.com/user-attachments/assets/be447abb-3b51-4c2a-ab87-0190f316271f)

เราใช้ Random Forests เอา Decision Tree มา Vote กัน 

![image](https://github.com/user-attachments/assets/5108de6a-9006-4f5a-ba24-e01e84a78448)

![image](https://github.com/user-attachments/assets/420fe10b-22b5-4169-a8b4-17fc9bdf5001)

* การเทรนด์โมเดล EfficientNet แล้วก็ Random Forest มาเทรนด์ต่อ


1. Freeze 3 Layer สุดท้ายก่อนให้มันเรียนรู้ 15 classes SoftMax

2. แล้วก็ Unfreeze ให้มากขึ้นโดยทำการ Fine-tuning

![image](https://github.com/user-attachments/assets/f951d081-f062-466b-b08f-c5ab4a2dc709)

## 3. Evaluation

![image](https://github.com/user-attachments/assets/83c9a44b-b82f-4281-b254-4ff88d3c2cde)

![image](https://github.com/user-attachments/assets/421433f3-c42c-4101-874e-d4e9514e429c)

## 4. Visualization

เราต้องการสื่อความหมาย สูงกว่า ต่ำกว่าให้กับคนใช้ทั่วไป User ทั่วไปที่ไม่ได้อยู่ในวงการ AI

Box-Plot
![image](https://github.com/user-attachments/assets/d651c7ba-e4a1-4d22-913c-0f7ac406ab39)

## 5. Future-works ของงานวิจัย

* สแกน CCA (มะเร็งน้ำดี) ตัวแรกของโลก
* 25 ความผิดปกติ ให้กับคนได้
* ถูกใช้ในโรงพยาบาลจริง 205 แห่ง
* AI ตัวนี้ยังอยู่ใน 150 GP's per year เอามาเทรนด์หมอต่อ
* Cloud-based AI Services (Deploy ขยายขนาดได้)

![image](https://github.com/user-attachments/assets/3cb15f3c-204d-4a09-bf8e-f36babd6766c)

ระบบ เราสามารถดูได้ว่าจะวินิจฉัยยังไง ก็จะสามารถดูได้

* Awards 

![image](https://github.com/user-attachments/assets/40efebe8-7389-43f6-9578-582c832d316f)

![image](https://github.com/user-attachments/assets/07b5cc97-5f3c-48ab-811f-348edadac3ac)

1. พบว่าถ้าเราทำใน Field Test มันลดลงทำให้งานต่อไปที่เราจะปรับปรุงโมเดล เราก็จะใช้ Prior knowledge สู.ขึ้น
เราจะเอาความรู้อื่นให้โมเดลมันเก่งขึ้น
![image](https://github.com/user-attachments/assets/96fc37e9-b6b2-4aa9-bcdb-b8c297f6c9ff)

2. ทำการเพิ่มจำนวนข้อมูล เพื่อให้โมเดลมีความแม่นยำมากขึ้น
![image](https://github.com/user-attachments/assets/6f7770a4-b6f8-433c-be90-dc2ff882dba7)





