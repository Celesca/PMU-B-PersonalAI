# Learning from Biosignal

Introduction :

What is Biosignal คือคลื่น/สัญญาณที่วัดได้ตามอวัยวะส่วนต่างๆของคน เช่น คลื่นไฟฟ้าสมอง หัวใจ
ปกติมีคนไข้เยอะมาก วินิจฉัยนาน เราสามารถใช้ AI Machine Learning มาวินิจฉัยขณะคนไข้อยู่ที่บ้านได้
นอกเหนือจากการสอบถาม และคลื่นไฟฟ้าสมอง สามารถนำแบบจำลอง Sleep Stage Scoring การวินิจฉัย Sleep Efficiency
ผ่านการวัดคลื่นสมอง หัวใจเป็นต้น เราจะทำเมื่อมีปัญหาการนอน เช่น นอนหลับไม่สนิท เป็นโรคหยุดหายใจ (Sleep Apnea) 

- ลักษณะของข้อมูลที่ใช้
- เตรียมข้อมูล
- ทำ CNN
- วัดผลประสิทธิภาพของโมเดล

## 1. Biosignal Analysis :

![image](https://github.com/user-attachments/assets/98281ca2-2fd4-439f-8bae-0c2d589ab4f1)

Application-Specific หมายถึง ตามอาการโรคที่เราสนใจ เช่นเราทำงานเกี่ยวกับโรคคลื่นหัวใจ เราก็จะต้องทำความสะอาดข้อมูลอีกรูปแบบหนึ่ง เช่นเดียวกับการ Feature extraction Pattern ของสมอง
หากเราเปลี่ยนคลื่นสองข้อแรกก็จะต้องเปลี่ยนเสมอ

เราพยายามใช้ Deep Learning มาช่วยในขั้นตอนสองอันแรก เพื่อสามารถเรียนรู้ Pattern หลากหลายได้
เขาใช้ Layer จากคลื่นสัญญาณให้กลายเป็น Vector ของอาการต่าง ๆ

![image](https://github.com/user-attachments/assets/8b98110a-eca4-4c05-97e5-f2d7b5ec279b)

## 2. Sleep Stage Scoring (การวิเคราะห์การนอน)

การวิเคราะห์การนอน คนไข้มีประสิทธิภาพมากแค่ไหน เราใช้ Sleep Efficiency
คนไข้ต้องมานอนในโรงพยาบาล เราจะเก็บสัญญาณเหล่านี้รวมว่า Polysomnogram (PSG) : EEG คลื่นสมอง, EOG จอตา , ECG คลื่นหัวใจ, EMG คลื่นกล้ามเนื้อที่คราง
ปกติเราเอาข้อมูลการนอนมา 6-8 ชั่วโมง

* Sleep Stage Scoring

ในแต่ละชั่วโมงเราจะติด Tag ทุกๆ 30 วินาที

![image](https://github.com/user-attachments/assets/abcb52ea-4610-4ec6-8192-5764c45d2aba)

Note : อันนี้ต้องยอมรับว่า ผมก็เคยทำ Research เกี่ยวกับด้าน Sleep Stage มาก่อนตอนสมัยปี 1 ครับ
* Non-rapid eye movement
N1 - Sleep onset พึ่งนอนหลับ
N2 - Light sleep หลับตื้น
N3 - Deep sleep หลับลึก
* Rapid eye movement (REM) - ฝัน
* Awake (W) ตื่นอยู่

ถ้าหาก Normal pattern คือมันจะวนเป็น Cycle หลับแล้วฝันจะปกติ ไม่ควรมีอันใดมากเกินไป เราต้องได้ครบทุกสเตจ แล้วก็ฝันยิ่งทำให้เรามีความคิดสร้างสรรค์
ซึ่งเราสามารถเปลี่ยนเป็น Multi-class classification problem in ML (Five Classes)

ตัวอย่าง คลื่นสัญญาณ PSG - Stage N2

![image](https://github.com/user-attachments/assets/105e5f89-9d44-4956-88dc-72d5629d44a8)

ถ้าคลื่น LOC ROC วัดจากจอตา, F C O วัดจากคลื่นสมอง ในช่วงเวลา 30 วินาที คุณก็ยังหลับตื้นอยู่

ตัวอย่าง PSG - Stage REM การฝัน (Rapid Eye Movement) แต่ในสมอง มันจะไม่ค่อยมี Activities

![image](https://github.com/user-attachments/assets/516e53a9-4342-4952-a308-de3936ccae8e)

การที่เรารู้แล้วว่า มี Stage ไหนบ้าง แล้วเราก็ไปคิดเลขและ Sleep Efficiency แทนที่จะต้องมานั่งดูเอง

![image](https://github.com/user-attachments/assets/7301073c-79fa-4ee1-8376-47ed30185b9e)

Problem ในปัจจุบัน เขาอยากให้ทดสอบได้ในที่บ้านเลย โดยใช้คลื่น EEG (จากคลื่นสมอง) แล้วถ้าใช้คลื่นสมองเราจะใช้ Deep Learning Model เพื่อให้คะแนนการนอนได้หรือไม่
ฉะนั้นเราจะได้ Wearable Devices ก็จะมีแบบใส่หู หรือผ้าคาดหัวที่มีอิเล็กโทรด

![image](https://github.com/user-attachments/assets/695758bf-124e-46cf-a5d5-967bc4796ef8)

* Public Sleep Dataset

เราจะใช้สัญญาณ EEG อย่างเดียว แล้วก็ทำ Label 5 Stage พร้อมทั้ง MOVEMENT, UNKNOWN
![image](https://github.com/user-attachments/assets/52720584-b40a-4d73-b3e3-3a78de8c6134)

## 3. Model ( DeepSleepNet 2017, TinySleepNet 2020 )

30s Single-Channel EEG จากนั้นผ่าน 1D CNN
* Representation Learning
* Sequence Residual Learning

ส่วนแรกประกอบไปด้วย CNN-Small มี Filter ขนาดเล็ก (มีระยะสั้น) / CNN-Large มี Filter ขนาดใหญ่ (มีระยะยาว)

แรงบันดาลการวิเคราะห์ Waveless Transform / Fast Fourier Transform

ถ้าอยากจะสกัดคุณลักษณะ Feature คลื่นความถี่สูงจะใช้ ขนาด Filter ขนาดเล็ก / คลื่นความถี่ต่ำ ต้องใช้ Filter ใหญ่

สมัยก่อน สมมติคลื่นสมองตรงตาม Pattern FFP มีคลื่นคล้ายๆ Sine Wave แต่ CNN มันจะเรียนรู้ด้วยตัวเขาเอง

![image](https://github.com/user-attachments/assets/8b93b56b-1cb7-4b12-b77d-74b7d8f19b4d)

CNN Large กล่องสีดำจะมีความยาวมากขึ้น ในแต่ละ Neuron Convolutional 1D มันจะมีการเปลี่ยนแปลงเสมอ
มันจะ Dot Product แล้วสไลด์ไปเรื่อยๆ 

![image](https://github.com/user-attachments/assets/8041b979-015c-49da-a346-852def35bb94)

สูงต่ำขึ้นมา เป้าหมายในการเรียนรู้ คือมี Pattern ไหนเกิดขึ้นใน 30 วินาทีนั้น เช่น Pattern ที่ Neuron ตัวนี้เรียนรู้เกิดขึ้น
ของตัวที่สอง Neuron ตัวที่ 2 มันจะมีทั้งหมด n ตัว แล้วเขาจะใช้ความรู้จากทุกๆ อันเพื่อแยกแยะว่าคลื่นนี้ควรอยู่ Stage ไหน

ต่อมา Sequence Residual Learning เราจะใช้ RNN

![image](https://github.com/user-attachments/assets/d9e04b9d-150d-420c-8268-308e4f0f7f97)

ปกคิ
