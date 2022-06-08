#include<ESP8266WiFi.h>
#include<ThingSpeak.h>
#include<SimpleDHT.h>
SimpleDHT11 x(2);
char s[] = "";
char p[] = "";
WiFiClient client;
unsigned long cn = ;
const char*API = "";
void setup() 
{
  pinMode(A0,INPUT);
  WiFi.begin(s,p);
  WiFi.mode(WIFI_STA);
  Serial.begin(9600);
  ThingSpeak.begin(client);
}
void loop() 
{
  int y = analogRead(A0);
  byte t;
  byte humidity = 0;
  int e = SimpleDHTErrSuccess;
  e = x.read(&t,&humidity,NULL);
  Serial.print(t);
  Serial.print(",");
  Serial.println(humidity);
  ThingSpeak.setField(1,int(t));
  ThingSpeak.setField(2,int(humidity));
  ThingSpeak.setField(3,y);
  ThingSpeak.writeFields(cn,API);
  delay(900);
}
