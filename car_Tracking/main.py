import cv2

cap = cv2.VideoCapture(r"C:\Users\Asus\PycharmProjects\car_Tracking\car.mp4")
fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=False) #gölgeler gözükmesin

while(1):
    ret,frame=cap.read() #videonun çerçeveleri frame değişkeni içinde kayıt edilecek
    fgmask=fgbg.apply(frame)  #çıkarma işlemi yapan maskeyi videoya uygulamak için apply
    median = cv2.medianBlur(fgmask,3)  #gürültü azaltmak için median filtresi

    #videodaki seçili reimleri renkli göstermek için
    (contours,hierarchy)=cv2.findContours(median.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    for c in contours:  #kontür aralığına göre dikdörtgen çizecek
        if cv2.contourArea(c) < 500:  #kontür aralığını belirliyoruz,her bir dikdörtgen için çizilecek
            continue
        (x,y,w,h) = cv2.boundingRect(c) #dikdörtgen aralığı
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),3)  #dikdörtgen çizmek için



    background = cv2.resize(median,(600,360))  #video boyutlarını ayarlamak için
    frame1 = cv2.resize(frame,(600,360))

    cv2.imshow('background',background)
    cv2.imshow('frame',frame1)

    k=cv2.waitKey(1) & 0xff  #videonun çerçeveleri arasındaki bekleme süresi. Parantezdeki değer azalırsa daha hızlı

    if k==27:  #esc ye basınca döngüden çıkacak
        break

cap.release()  #video oynatması sona eriyor
cv2.destroyAllWindows()  #tüm pencereler kapanır




