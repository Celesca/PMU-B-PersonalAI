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

3.1 DeepSleepNet
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

เราให้โมเดลเลือกได้ว่า จะเอา Sleep Stage ใส่เข้าไปในโมเดลได้ มันเลือกได้ว่าจะเอาความรู้จาก RNN ไปใส่เข้าไปเท่าไหร่ เพราะได้ Shortcut มาจาก CNN

รายละเอียดในการสอน (Learning Algorithm)
Weight (Filter) มันก็คือ Filter ที่โมเดลเรียนรู้ไปเรื่อยๆ ตัวสีแดงก็จะถูกปรับเปลี่ยน เราอยากได้ Filter หรือ Pattern ที่ดีที่สุด

![image](https://github.com/user-attachments/assets/d0b76996-529a-4a42-8a69-58e20b3bd603)

ถ้าหาก x(i) ตรงกับ Label y ก็จะทำให้ค่า Loss เป็น 0
โมเดลเราอยากได้ Theta มีผลรวมของ argmin น้อยที่สุด

3.2 Tiny SleepNet

![image](https://github.com/user-attachments/assets/6d30e7b0-3ede-4d2f-93de-55e02de109d9)


Representation Learning ยังใช้ CNNs เราตั้งสมมติฐาน เราตั้ง Filter ที่มีความถี่สูงต่ำ โดยที่ไม่ต้องแยก Branch ออกมาจะทำให้โมเดลมีขนาดเล็กลง โดยที่ x(i) ยังเป็น 30 วินาทีอยู่

![image](https://github.com/user-attachments/assets/6183156b-f4b3-48e6-8b6b-b2fbc5da9bf7)

Sequence Learning จะใช้ RNNs แต่มีความเล็กที่น้อยลง แทนที่จะมี RNN สองตัว แต่เราใช้เป็นตัวเดียวแทน

![image](https://github.com/user-attachments/assets/0b0612cf-482f-4509-afa4-0272fc1e9c91)

TinylSleepNet ถูกบีบเล็กลง สอนก็จะมี Parameter ที่มีอยู่อย่างจำกัด เราจะใช้ Data Augmentation 

* Weighted cross-entropy loss มันจะไม่ Overfit เช่น Label มัน Imbalance datasets เช่น Sleep Stage
เราลงโทษ N1 มากกว่าเวลาเราทำเป็น Stage N2 N3
มันไม่มี Pretraining set ไม่ต้องแบ่งส่วน ไม่ต้อง Finetune ข้อมูลก่อน มันทำให้มันสอนง่ายขึ้น ถ้าใช้โมเดลเองนอกจากโมเดลขนาดเล็ก มันก็เทรนด์ง่ายขึ้น

* Signal Augmentation -> คุณหมออาจจะแบ่ง 30 วินาทีบนคนเดียวกันอาจจะแตกต่างกันออกไปได้
แรงบันดาลใจคือ การเลื่อนหรือถอยไปข้างหน้านิดหน่อย เราก็จะมีรูปแบบเยอะมากขึ้น โดยทำการ Shift แล้วทำให้เราได้ Data ที่มากขึ้น ทำให้เกิด Variation เราก็จะเลื่อนแบบ Random

![image](https://github.com/user-attachments/assets/49676d48-e181-4666-bfe9-6ba2ba193857)

* Sequence Augmentation -> ไอเดียคือถ้าเราเจอคลื่นการนอนแบบเดิมเสมอในคนไข้คนเดิม
เราก็เลยคิดว่า เราจะข้ามไปบ้างเพื่อให้มีความหลากหลายมากยิ่งขึ้น เช่น S1 โดนสุ่มให้ไม่ต้องข้าม S4 ข้ามไป 3 มันทำให้เราสุ่มข้ามแล้วเพิ่มความหลากหลายให้กับข้อมูล

![image](https://github.com/user-attachments/assets/17949a7a-4b71-4410-81f4-5ca29f72e84f)

4. Model Evaluation

![image](https://github.com/user-attachments/assets/201e2006-cec9-4f4a-9947-5dc406122174)

การแบ่งข้อมูลใช้ K-Fold Cross-Validation (การแบ่งข้อมูลจะไม่มีคนไข้คนเดียวกันอยู่ใน Train ซ้ำกัน)

Accuracy, MF1, Cohen's Kappa -> วัดผลโดยรวมของ Stage ทั้ง 5

ของแต่ละ Stage ให้ใช้ ACC, Precision Recall, และก็ Hypnogram

* 5-Fold Cross Validation

![image](https://github.com/user-attachments/assets/b363af59-15a3-4f53-a669-e0111bc93852)

Test dataset อาจจะเอาข้อมูลอื่นที่ไม่ใช่ Training set มาลองทดสอบ

* Confusion Matrix

![image](https://github.com/user-attachments/assets/4c8f6b0a-6924-4822-a22c-b3a1acd74981)

สำหรับ Task เราสามารถทำนายจากการที่ N2-Stage กี่ชั่วโมง แล้วโมเดลทำนายได้กี่ชั่วโมง

![image](https://github.com/user-attachments/assets/339d0bef-b59f-442e-ab45-0b21b36ce1fb)

* Hypnogram

![image](https://github.com/user-attachments/assets/4f5c63da-e925-4e93-92e7-ca6c8169ceaf)














