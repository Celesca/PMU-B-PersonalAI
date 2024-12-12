# AI for arrest criminals


## 1. Object Detection

Object Detection คือการ Classification and localization โดยที่ได้ Bounding box ขึ้นมา

เราต้องคำนึงถึงความเร็วแบบ Real-time และ High accuracy

![image](https://github.com/user-attachments/assets/bd5012c8-40b8-45e7-a6f5-d84418b97984)

ความสำคัญของ Object Detection

![image](https://github.com/user-attachments/assets/5fc29574-51fb-408c-a213-7a8d8354ef2c)

ยกตัวอย่างเช่น YOLO จะเอาเป็นรูปมา หรือว่าจะเอาเป็นเฟรม ซึ่งเป็นเรื่องของ Visual ซึ่งสำคัญมากในงานของ Object Detection
Computer Vision & Deep Learning ที่ทำให้มัน Real-time ได้

![image](https://github.com/user-attachments/assets/b8ea4d8e-e9c9-4f2b-b625-682413081410)

เป็น Unified Realtime Object Detection -> แบ่งภาพออกเป็น Grid โดยที่ถ้า Sensor ไปอยู่ใน Grid Cell มันก็จะมีหน้าที่เกี่ยวกับ Detect Object นั้น

v1 - ใช้ Framework Darkness

V2 - ใช้ Ankle Box ได้รับการพัฒนา Detect ได้มากกว่า 9000 Class เทรนด์กับ ImageNet และ BOC Datasets และ Introduce Multiscaled

v3 - Ankle box มันสามารถที่จะ Detect Object ที่มีขนาดที่แตกต่างกันได้ดีขึ้น โดย Scaled การดีเทค 3 แบบ แล้วก็ใช้ Logistic Regression ที่ output แทน Softmax

v4 (2020) - Speed & Accuracy CIOU Loss, Planet, SIM, Darkness Function

v5 - ไม่ใช่ Official Version ของผู้พัฒนา แต่เป็นตัว Improved มาเรื่องความเร็วและ Performance , Reference Time, จัดการขนาดเล็กจนใหญ่ได้ดีขึ้น ใช้ Leaky ReLu Activation

v4tiny - การรันแบบ real-time processing กับอุปกรณ์ที่จำกัดทรัพยากร

YOLOR - Transformer Mechanism การทำ Generalization ของ Detector
YOLOX - บริษัท Ankle Preprocess

YOLOv7 - โฟกัส Small Optimzation YOLOv6 - การทำ Quantization

YOLOv8 and NAS - เป็นเวอร์ชั่นที่ให้ Performance Accuracy ที่สูงขึ้นอย่างมีนัยสำคัญ และมีการเทรนด์จาก COCO ที่เน้น Accuracy เพิ่มขึ้น

## 2. YOLO Overview

![image](https://github.com/user-attachments/assets/84e86dfe-b521-44fd-a103-f70c71292af6)

Size ของ Grid cell และแต่ละ Cell จะทำการ Detect Object แล้วก็ Center เพื่อช่วยในการระบุ Object ที่อยู่ในภาพ หลังจากการ Split Image
มันก็จะสร้าง Bounding Boxes (B) เราจะ Predict Bounding Boxes เหมือนคลุม Correlate ของบล็อคที่มีอยู่ใน Image นั้นๆ
Bounding box Prediction - ในแต่ละบล็อค มันจะมี Object ที่ตกอยู่ในนั้นๆ โดย Predict จะคำนวณ 5 ตัวได้แก่ x y w h confidence

x, y -> center of bounding boxes ของ Grid cell เราก็จะได้ 0.5 0.5 เพราะว่า ซ้ายเป็น 0 และ 1 เป็นขวา

w, h -> Represent ขนาดของ Bounding Boxes เช่นถ้ามีค่าเป็น 1 มันก็จะมี Span เป็น 1 ของภาพนี้

Confidence -> ระบุว่า Bounding Boxes ของเรามัน Accuracy เท่าไหน ที่มันสามารถ Detect ได้อย่างถูกต้องแล้ว เราจะใช้ IOU Intersection and Union

ถ้าค่า IOU = 1 มันได้ Overlap กับ Ground Truth ได้อย่างถูกต้อง แต่ถ้าเป็น IOU = 0 มันก็จะไม่มี Overlapping เลย

ฉะนั้นถ้าไม่มีการ Detect ใน Grid Cell แสดงว่า Confidence ก็จะเป็น 0

Detect ได้รวดเร็ว โดยเฉพาะในเรื่องของ Performance ภาพรวม

### Training YOLO

YOLO เป็น Regression Algorithm -> สมัยก่อนจะใช้ Traditional เช่น อาจจะทำ Classification ทำ Post processing แล้วก็ลบ Duplicate processing

แต่มันทำงานใน Single Regression algorithm มันจะให้ผลจากภาพโดยตรง เราก็เลยตั้งชื่อว่า YOLO มันจะมีตัวแปรที่เกี่ยวข้อง


ในการเทรนด์โยโล่ เราต้องเข้าใจ X, Y คืออะไร

![image](https://github.com/user-attachments/assets/12e4f92a-dcbd-4f80-b416-258aa17eacf6)

Output ที่ได้จะเป็น Tensor มีขนาดโดยจะ Split เป็นขนาด S x S Grid cells สำหรับการ Predict Bounding Boxes ที่จุด Center วางอยู่ในนั้น

B -> จำนวน Bounding Boxes ที่ Grid Cell Predict ออกมาได้ เพราะฉะนั้นในแต่ละ Bounding Boxes จะมี 5 ค่า


## 3. YOLO Architecture

![image](https://github.com/user-attachments/assets/f7f91224-3a24-47ef-88ec-eeca6c23c8ee)

สมมติว่าเรามีขนาด 448 x 448 x 3 ช่องสี และประกอบไปด้วย 7 Conv Layer เหมาะสมในการจัดการ Image Data มันสามารถ Capture ข้อมูลพื้นที่ได้ดีสำหรับ CNN

ในแต่ละ Layer จะมีการ Detect Pattern Features คุณลักษณะสำคัญต่าง ๆ มันจะไล่จากการ Detect Basic ก่อนเช่น Simple Edges ในเลเยอร์เบื้องต้น ยิ่งในเลเยอร์ท้ายๆ ก็จะมีการ Extract Feature ที่มีความ Complex มากยิ่งขึ้น ในส่วนของ CNN

การใช้ Filter และ Activation Function (S = 7 ถูกแบ่งเป็น Grid 7x7) แล้วก็ B = 2 ในแต่ละ Grid Cell จะ Predict Bounding Boxes ก็จะมีเรื่องของ Dimension มันก็จะ Includes ค่า Confidence Score มันสามารถ Detect Object ใน Box ได้มากแค่ไหน

C = 20 สำหรับคลาสที่โมเดลจะดีเทคได้

* เคสของ Multiple Bounding Boxes ที่ทับซ้อนกัน ให้ตัดอันที่ไม่จำเป็นออก ที่ไม่ได้มี Object นั้นอยู่ดีๆ

![image](https://github.com/user-attachments/assets/287e3c93-bc72-424d-871e-636154a1f3de)

เราตั้งค่า Threshold ว่า Confidence Score มันถึงรึเปล่า Filter Bounding Boxes ที่ต่ำกว่าที่เราตั้งไว้ เราพูดง่ายๆ Confidence Score ต่ำๆ ออกไป แล้วก็จะ Identify Class ก็จะต้องมีการคำนวณ Class Score เราก็จะทำ Estimate ตัวโมเดลได้
โดยเขียนเป็น Distribution ว่า Bounding Boxes นั้นเป็น Class ไหน ฉะนั้น "Non-maximal suppression" ก็จะตัด Redundant Bounding Boxes ที่มี Object นั้นอยู่จริงๆ ช่วยให้แม่นยำ ช่วยในการตีความได้ดีขึ้น

YOLO Prediction พอเราลด Redundant ก็จะช่วยในเรื่องของ Prediction พอเราใช้ Non-maximal suppression หรือเราเน้น mAP (Mean Average Precision) อาจจะเพิ่มได้ 2-3% จากไฟนอล แต่ว่ามันก็มีประสิทธิภาพสร้างความแตกต่างได้ใน Final Output

เราต้อง Finalize ว่า Make sure ว่ามันมี Object ใน Bouding Boxes นั้นได้

ฉะนั้นตั้งแต่ทำ Redundant, mAP score ก็จะช่วย Performance โดยรวมของระบบ หากเราต้องใช้ Real-time

![image](https://github.com/user-attachments/assets/f440f8fe-89f4-409e-972a-e877f89b2349)

## 4. YOLO Objective Function

การที่เรารู้ Objective Function จะช่วยให้เรา Optimize ค่าได้ ต้องทำความเข้าใจ

![image](https://github.com/user-attachments/assets/423402dc-350a-45ab-ad59-f8cc8e6f00af)

มีทั้งหมดสามฟังก์ชัน ก็คือ Localization Loss จะคำนึงถึง Position
Ensure ว่า Bounding Boxes ที่ได้มาจากการ Predict มัน match กับ Bounding Boxes ที่เป็น Ground Truth มันก็จะทำงานกับการหา Error ระหว่าง
Predicted Bounding Boxes คำนึงถึง Position and Size ซึ่งจะเป็นส่วนของ x,y coordinates

penalized coordinated จะใช้ mean squared error โดยการยุ่งเกี่ยวก็จะมี Ground Truth bbox in the ith cell โดยที่ ground truth

พจน์บน จะดู Coordinate X,Y ในเชิงของ Position

พจน์ล่าง ดู Size ความกว้างและความสูง โดยที่จะใช้ Square Roots ให้กับความกว้างและความสูง โดยที่เราอยากให้ค่า Loss มันน้อยๆ เมื่อเทียบกับ Ground Truth ค่า Error มันต่ำ

คือเราอยากให้ผลลัพธ์มันไกลเคียงกับ Ground truth มากสุด เราก็จะทำ Optimize ให้มีค่าที่ต่ำ

Confidence Loss - สำหรับการ Predicted Bbox

Classification Loss - เปรียบเทียบระหว่าง Ground truth ของคลาส c ที่อยู่ใน cell กับ predicted conditional
![image](https://github.com/user-attachments/assets/866c580e-89cc-43e4-afd9-0b7ef770af0c)

YOLOv8 เปลี่ยน CSplayer เราจะเรียก C2f module หรือเรียกว่า 2 Convolution รวม High-level features with contextual information เพิ่มประสิทธิภาพ
![image](https://github.com/user-attachments/assets/a591223c-d018-4f2e-8063-ebc7a802ef58)

![image](https://github.com/user-attachments/assets/0e5e783c-b3a0-424c-9cfa-1d3407b9cdd9)

Head และ Detail ภายใน มี Conv Split Bottleneck Conv ที่จะเพิ่มเติมต่อจากภาพนี้

![image](https://github.com/user-attachments/assets/a2046625-0cfa-4024-b1c7-ba6b51e57f94)

เราจะ Replace C3 Module ด้วย C2f module เข้ามาแทนที่ แล้วก็จะ replace convolutional layer จะใช้ 6x6 แต่ v8 จะใช้ 3x3 in backbone
แล้วก็ลบไปสองลำดับ ลำดับที่ 10 และ 14 (Improves ลดความซับซ้อนและเพิ่มความแม่นยำ)

First Layer -> เอา 3x3 เข้าไปแทนใน Bottleneck ต่างๆ เพื่อเพิ่มประสิทธิภาพให้กับ YOLO

Datasets Coco และ Roboflow ถ้าเทียบกับเวอร์ชั่นอื่น แล้วก็เรื่องของ Community มีคนใช้เยอะ ฉะนั้นสะดวกใช้งาน



