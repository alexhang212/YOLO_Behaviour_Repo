.. _yologuide:



Is YOLO-Behaviour appropriate for my study system?
=====================
For any new computer vision method, it is important to figure out whether it is appropriate for your study system. While YOLO-Behaviour is super powerful and robust, it has a few major shortcomings that make it less appropriate for certain study systems compared to others. Here are a list of considerations you can think about before diving into a project and spending time annotating data:

1. Is the behaviour you want to detect visually distinguishable from a single image?
2. Do you have sufficient images per behavioural class to train the model?
3. Are the videos relatively consistent/ without major noise?
4. Do you have (or can prepare) a validation dataset to ensure the model is detecting event appropriately?

If the answers to the questions above are all YES, then you should be good to go. If not, I expanded on each point below, to help figure out whether this method is appropriate for your study system.

|start-h1| 1. Is the behaviour visually distinguishable? |end-h1|
This is the most important thing to consider, and the major shortcoming of the method. The YOLO model only analyzes **SINGLE IMAGES**, and tries to detect a behaviour frame by frame. If the behaviour you want to detect has a temporal component, or is context dependent, or if even you cant tell from an image whether a behaviour is happening, then the model will likely also fail.

For example, let's say you would like to detect a pigeon spinning around as part of its courtship display. From the video, you will be able to see it turn around, but for YOLO, since it only takes in single frame inputs, every frame of this spinning behaviour will only look like a pigeon standing still or walking. This would be an example where **temporal information** is important, and YOLO-Behaviour might not be appropriate. For this task, methods that takes in video input (like DeepEthogram) or methods that first does keypoint annotation might be more appropriate.

As a second example, let's say you want to detect whether Siberian Jays are doing a submissive display or flying (This was a real example that was excluded in the manuscript because the method was not appropriate). The submissive display is a behaviour where the jay flaps their wing rapidly while screaming when sitting on the perch. In this case, this behaviour is very visually similar to the birds flying, and was easily confused with birds flying in and out of the frame. This is an example where the **context matters**, since which behaviour it is depends on what the bird is doing before and after this behaviour, or other cues (like audio).

Hopefully these examples can give an idea of when the method might not work. Just remember all the model has is the single frame, and any **visual features** of the image. So anything that cannot be determined from the image alone, will likely not work.

|start-h1| 2. Do you have sufficient images per behavioural class? |end-h1|
For any computer vision task, you need training data. YOLO-Behaviour is the same, with the requirement of around 1000-1500 frames for each case study explored in the manuscript. As a result, if you have a relatively rare behaviour (that only happened a few times), the model will struggle to learn. The rule of thumb is to have at least a 200-300 frames per behaviour, with a variation in contexts (not super similar frames). A lot of times, obtaining and choosing these frames might take more time than annotating the bounding boxes itself.

|start-h1| 3. Are the videos relatively consistent/ without major noise? |end-h1|
I put this here, but this is a general rule for applying computer vision tools. Is your training data relatively similar to the final videos where you want to apply the model to? For example, if you trained a detector that detects chimpanzees in a zoo, you cannot expect the model to also work with chimpanzees in the jungle. For YOLO-Behaviour, the videos should be as clear as possible, having the subjects of interest in focus, with reasonable resolution. Computer vision tools are powerful, but it's not magic. So ensure the data quality is of a certain level of standard, usually from common sense. Examples of bad quality videos include: super distorted videos; or subject of interest is blurry; or thermal drone footage where the subject of interest is 2 pixels wide; or underwater videos where waves or bubbles occlude the subjects most of the time etc. etc. 

|start-h1| 4. Do you have a validation dataset ready? |end-h1|
Finally, while it can be annoying, any new method applied to a new system should be validated to ensure no bias is introduced. For YOLO-Behaviour, in addition to the image training data, it is important to also prepare a validation dataset of a given amount of videos where you annotate the ground truth events (with tools like BORIS), to ensure the whole pipeline is working well. How much validation is needed will depend on what is "good enough" for your application, but I would make sure some validation is done. The accuracy of these models can be misleading when visualizing it on video (usually it looks good on video), but quantitative evaluation is essential.


.. |start-h1| raw:: html

     <h1>

.. |end-h1| raw:: html

     </h1>