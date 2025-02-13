Validation and grid search
===========================

Here, I will introduce the validation and grid search workflows. 
This will be different for each study system, and will depend on the format of the manual annotation you have. 

This page would be slightly different from the other pages in the documentation, as I will give general guidance on generating evaluation metrics within python, so that you can readily adopt this to your own system.
As with other pages, I will provide sample code for the Siberian Jay system.

|start-h1| Running the sample code |end-h1|

If you would just like to try out the sample code, here is how you can run it for the Siberian jay system:

.. code-block:: python

    python Code/6_SampleValidation.py


Now, I will try to breakdown the validation procedure using the sample script for the Siberian Jay dataset.
The overall aim is to create two python lists, one as the ground truth, and one as the predicted classes. 
Then, we can use the `sklearn` library to calculate all kinds of evaluation metrics.
How the ground truth and predicted classes are matched up will depend on the format of the manual annotation you have, that's why this step will require some data wrangling.


Here, I defined the two empty lists before looping through the videos to get predictions.
Throughout the script, I will then populate these lists with the ground truth and predicted classes.

.. literalinclude:: 6_SampleValidation.py
   :lines: 24-34
   :linenos:
   :lineno-start: 24
   :language: python
   :emphasize-lines: 5-6

Then for each video, I loaded the detection pickle file (See :ref:`inference` for more details), then used the SORT tracker to get behavioural events.

.. literalinclude:: 6_SampleValidation.py
   :lines: 35-71
   :linenos:
   :lineno-start: 35
   :language: python

This results in a list of bounding boxes and tracking IDs for each frame in the video. We will use this list to match up with the ground truth later.

Next, we will load the manual annotation from BORIS. 
The BORIS data is annotated as absolute time, so we need to multiply it by the frame rate (25 here) to get back frame number.
We then populate a list with the same length as the predicted list, then count the number of individuals "feeding" for a given frame

.. literalinclude:: 6_SampleValidation.py
   :lines: 74-86
   :linenos:
   :lineno-start: 74


Next, we will match up the ground truth and predicted classes, but with a certain time window.
This would be different for different kind of data, we refer to the original publication for discussion on this.
For the typical cases, perhaps you only have to match up whether a predicted class was correctly identifying an event, but here because of the slight mismatch in manual annotation and the automated method, we need to summarize datapoints into time windows.

.. literalinclude:: 6_SampleValidation.py
   :lines: 88-96
   :linenos:
   :lineno-start: 88

Finally, we will loop through the time window lists and populate the global prediction and ground truth lists.

.. literalinclude:: 6_SampleValidation.py
   :lines: 97-114
   :linenos:
   :lineno-start: 97


Now with the ground truth and predicted classes, we can use the `sklearn` library to calculate all kinds of evaluation metrics.

.. literalinclude:: 6_SampleValidation.py
   :lines: 116-150
   :linenos:
   :lineno-start: 116

The code above will calculate the precision, recall, F1 score, and confusion matrix for the two lists of predicted and ground truth classes.


|start-h1| Grid search |end-h1|

After figuring out the validation pipeline and obtaining metrics, you can then proceed to grid search to optimize hyperparameters of the SORT tracker.
This step is not strictly necessary, but is just a standardize way of making sure everything is optimized for your system.

You can run this in the terminal for the Siberian jay dataset:

.. code-block:: python

    python Code/7_SampleGridSearch.py


The core of the script is essentially the same as the validation script, but the difference is that here, we loop through the validation a lot of times to test out all cominbations of hyperparameters.

.. literalinclude:: 7_SampleGridSearch.py
   :lines: 151-158
   :linenos:
   :lineno-start: 151

Here is where we define the range of parameters we want to explore in the ``Code/7_SampleGridSearch.py`` script.
The script will then loop through the combinations of these paramters and save it as a csv.

We do note that the validation function can be essentially the same as the one used for validation above, except for a small change at the end to save the validation metrics instead of printing it.

.. literalinclude:: 7_SampleGridSearch.py
   :lines: 124-129
   :linenos:
   :lineno-start: 124

After running the script, the results will be saved in a csv file, then you can choose your best hyper parameters!

Hope this whole workflow was clear, and here is the whole YOLO-Behaviour pipeline! If you have any questions, feel free to contact me at ``hoi-hang.chan[at]uni-konstanz.de``


.. |start-h1| raw:: html

     <h1>

.. |end-h1| raw:: html

     </h1>