# LEGO Vision Inspection – Speaker notes

## 1. LEGO Vision Inspection

This LEGO part has one tiny mistake. A human inspector might miss it at the end of an eight-hour shift. This camera finds it in about 40 milliseconds, and there is no AI model behind it. Today I'll show you exactly how assembly lines check parts with a camera, and you can build it yourself with a webcam and a few hundred lines of Python.

## 2. A missing part costs far more than the part

In a real factory, a missing part that reaches the customer can mean a recall, a line stop, or a lost contract. Human inspectors get tired; cameras don't. Industrial vision sensors do exactly this job, but they are expensive. Today we use a regular HD webcam and the same principles. [Add a short real example from your own factory here.]

## 3. Seven parts, one simple rule

Our product has seven parts: one orange base, four transparent red studs, one black 2 by 2 brick, and one lime pin. The rule is simple: red on all four corners, black in the center, lime on top of the black brick. This parts list is our bill of materials, the BOM.

## 4. Two cameras guard the assembly

The system has two inspection points, just like a real line. Station one is a kitting check: are all the parts in the tray before assembly starts? Station two checks the finished product: is every part in the right place? Each station gives a clear OK or NG, and an NG always says why.

## 5. Lighting does most of the work

In machine vision, lighting is most of the job. Camera straight down, a matte white background, and light from both sides so there are no hard shadows. If you can, lock the camera's auto exposure and white balance, otherwise the colors drift when the light changes. [Show the setup from the side with your phone here.]

## 6. Five steps from pixels to a decision

Every inspection runs the same five steps. First, everything that is not the white background is treated as an object. Second, each object or window is classified by color in HSV space. Third, we count the parts or check each position. Fourth, we compare with what we expect. Fifth, we decide: OK, or NG with the reason. No training data and no AI model, just classic machine vision.

## 7. Count by color, compare with the BOM

Station one asks one question: is everything in the tray? The program finds every object, names its color, and counts per color. Then it simply compares with the bill of materials. In this example, one red stud is missing and a foreign part fell in, so the result is NG with both reasons. One limitation: touching parts are counted as one object. That's why real kitting trays have separate pockets.

## 8. Check 16 studs and one pin

After assembly, counting is not enough: a part can be present but in the wrong place. So the program finds the base, straightens it with a perspective transform, and checks sixteen stud positions plus the pin. Every circle is a small inspection window, the same idea as the position tools in industrial sensors. If the operator puts the part in rotated by 180 degrees, it still passes, because both orientations are checked. [Press g to toggle the grid live.]

## 9. Every NG says exactly what is wrong

Here a red stud was moved from the corner to the stud next to it. The program marks both windows red: red missing at the top-left corner, and red misplaced at row one, column two. Every NG names the problem, so the operator knows what to fix. We will try all of these live in a moment.

## 10. Look only where the part should be

A wide camera view sees cables, hands and table edges. So we inspect only inside a square in the middle. Everything outside is dimmed and ignored, and objects touching the edge of the square are skipped too. In this test, the blue and red blocks and the dark cable outside the box did not change the result. Industrial sensors call this the region of interest.

## 11. Let's break it on purpose

Switch to the live program now. Station one: press 1. Show the full kit, then remove one red stud, then add a foreign part. Station two: press 2. Show the correct part, then each defect one by one, and finally rotate the part 180 degrees. Optional: switch off one light to show a false NG caused by a shadow. Press m for the color mask view, g for the grid, f for full screen.

## 12. Classic vision is fast, but it trusts color and light

Classic vision is fast and needs no training, but it depends on color and lighting. A shadow can turn red into almost black. In the next episode we solve the same problem with a trained YOLO model on a Jetson, and then compare it with an AI vision model. Fill in the measured speeds after those tests; the brackets are placeholders.

## 13. The full code is free on GitHub

The full code is on GitHub, link in the description. It's MIT licensed, and the sample photos are included, so you can try it without a camera. Next episode: the same line with a trained YOLO model on a Jetson. Subscribe so you don't miss it. Thanks for watching.
