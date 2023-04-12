# CSE 404 Project Group 11


## Authors

- [Austin LeBlanc](https://github.com/austinleblanc)
- [Dylan Burke](https://github.com/DylanDBurke26)
- [Ryan Kunkel](https://github.com/rykunk21)
- [Bilal Pargan](https://github.com/parganbi)


## Writeups
### Austin LeBlanc
 My model utilizes 4 LSTM layers with 55 units and 3 dropout layers.
 The model's evaluation losses are as follows: 
 1. 0.013708480633795261
 2. 0.0
 3. 0.013286259956657887
 4. 0.003933305386453867
 5. 0.0
 6. 0.00748058408498764
 7. 0.000450088526122272
 8. 0.005716415122151375
 9. 0.006507860030978918
 10. 0.006938012316823006
 11. 0.00706125795841217
 12. 0.00046757544623687863
 13. 0.003627857193350792
 14. 0.0058596935123205185
 15. 0.011294116266071796
 16. 0.0007059662020765245
 17. 0.0014749204274266958
 18. 0.008419246412813663
 19. 0.0019708387553691864
 20. 0.007465030532330275
 21. 0.013262602500617504
 22. 0.008780421689152718
 23. 0.0047981939278542995
 24. 0.00791484396904707
 25. 0.005592113360762596
 26. 0.0026013560127466917
 27. 0.00751833338290453
 28. 0.004583251662552357
 29. 0.005393284372985363
 30. 0.005929749924689531
 31. 0.0057762362994253635
 32. 0.00627580750733614
 33. 0.007136096712201834
 34. 0.004738216754049063
 * Each of these entries correspond to a product family, in chronological order, with the 34th entry being the loss of total sales. The prediction on the test data for each family is visible in austin_model.ipynb. 
 * My training losses during the final epoch of each family's model fitting are as follows:  
 1.  0.0133
 2.  0.0000e+00
 3.  0.0128
 4.  0.0043
 5.  0.0000e+00
 6.  0.0081
 7.  4.8157e-04
 8.  0.0058
 9.  0.0072
 10. 0.0077
 11. 0.0074
 12. 5.3808e-04
 13. 0.0033
 14. 0.0060
 15. 0.0110
 16. 3.8297e-04
 17. 0.0015
 18. 0.0091
 19. 0.0020
 20. 0.0073
 21. 0.0135
 22. 0.0096
 23. 0.0052
 24. 0.0065
 25. 0.0059
 26. 0.0025
 27. 0.0073
 28. 0.0041
 29. 0.0056
 30. 0.0064
 31. 0.0062
 32. 0.0049
 33. 0.0075
 34. 0.0058
 * Full output results are visible within the notebook's outputs.
 * Potential improvements could come from dropping zero-valued data from the dataset, as well as potentially examining seasonality, or incorporating the provided oil data into our predictions. 


### Bilal Pargan
I created a 2 layer LSTM model. The model contains 140 units, 0.00001 learning rate and linear activation model.

When training, I trained my model on 4 different stores. The following losses are:
store#1 - 0.0378
store#3 - 0.0432
store#10 - 0.0201
store#17 - 0.307

After training on these stores, I wanted to see how my model will fit to a store we did not train it on. When looking at the figure on the bottom of bilal_model.ipynb, we can see that the model captures the general trends, however, it regularly fails to reach the highest and lowest sales values, falling just short. This could be an issue in how I am normalizing my data but I am unsure on that front. 

In general, I think my model does a good job at capturing general trends, as when the actual sales fall, so do the predicted sales and vice versa for when sales rise.

### Aron Tobias
I decided to undertake the uninteresting but necessary task of providing some sort of control test.
In particular, I tried to get results better than what my teammates have using just one LSTM layer.
My notebook is largely derived from Austin's work, to ease data collection and training routines. This is with his permission.

Losses:
- Store 1 Auto: .0135
- Store 1 Beauty: .0393
- Store 1 Beverages: .0030
- Store 1 Books: 0.0

STEP 6:
My model was SARIMA, which can be found in the SARIMA folder with an associated README which is a little more in-depth.
For losses and metrics, consult the notebook. AIC is apparently a good metric?


# Dylan Burke
I attempted to train a model to predict the total sales for each day for a store given the product families. I split the data for each store into:
* 60% train
* 20% Validation
* 20% Test

The shape of the training and validation data is
* The day of the sales
* The number of look back days
* The sales of the product family

The shape of the testing data is
* The day of the sales
* The total sales of all the product famlies


My model consists of: 
* 2 LSTM layers with a variable number of units with input shapes of of the
* A dropout layer with a value of 0.2
* A final dense layer with 1 ouput
``` python
def train_model(train_x, train_y, val_x, val_y, num_lstm_units, num_epochs, batch_size, test_all):
    model = Sequential()

    if test_all:
        model.add(LSTM(units=num_lstm_units, return_sequences=True, input_shape = (train_x.shape[1], train_x.shape[2])))

    if not test_all:
        model.add(LSTM(units=num_lstm_units, return_sequences=True, input_shape = (train_x.shape[1], 1)))

    model.add(LSTM(units=num_lstm_units))
    model.add(Dropout(0.2))

    model.add(Dense(units=1))

    loss_fn = tf.keras.losses.MeanSquaredLogarithmicError()
    model.compile(optimizer='adam', loss=loss_fn)

    hist = model.fit(train_x, train_y, epochs=num_epochs, batch_size=batch_size, validation_data=(val_x, val_y))

    return model, hist
```
I trained one model per store using the product families sales to predict the total_sale value for each day of data

The final losses and validation loess for each store are:
<pre>Loss, Validation Loss
1: 0.00598, 0.01098
2: 0.0028, 0.07952
3: 0.00343, 0.00446
4: 0.00308, 0.00514
5: 0.00272, 0.00354
6: 0.00292, 0.00448
7: 0.00299, 0.00455
8: 0.00339, 0.00907
9: 0.00323, 0.00589
10: 0.00508, 0.01822
11: 0.00401, 0.00472
12: 0.00221, 0.00729
13: 0.00382, 0.00851
14: 0.00582, 0.00791
15: 0.00257, 0.00837
16: 0.0035, 0.00957
17: 0.0024, 0.00705
18: 0.00366, 0.00982
19: 0.00387, 0.00578
20: 0.00108, 0.04796
21: 0.00056, 0.0085
22: 0.00048, 0.09057
23: 0.00389, 0.00417
24: 0.00359, 0.01406
25: 0.00216, 0.00538
26: 0.00109, 0.00396
27: 0.00322, 0.02566
28: 0.00404, 0.00901
29: 0.00136, 0.01187
30: 0.00352, 0.00699
31: 0.00137, 0.01719
32: 0.00341, 0.00474
33: 0.00402, 0.00962
34: 0.00244, 0.01155
35: 0.00066, 0.00201
36: 0.00335, 0.00934
37: 0.00352, 0.00445
38: 0.0036 , 0.00715
39: 0.00421, 0.00666
40: 0.00312, 0.00592
41: 0.00261, 0.01228
42: 0.00061, 0.00571
43: 0.00237, 0.00548
44: 0.00396, 0.00601
45: 0.0036, 0.01128
46: 0.00333, 0.00547
47: 0.00328, 0.00496
48: 0.00292, 0.00585
49: 0.00317, 0.00947
50: 0.0027, 0.00811
51: 0.00369, 0.00554
52: 0.0, 0.0
53: 0.00269, 0.00642
54: 0.00257, 0.00447
</pre>
Some of the losses are close to each-other, but others are pretty far from eachother.
This gap could be closed by better fine tuning the parameters of the model, and possibly changing the size and dimensions of the training and validation sets.


A graph of each stores losses vs validation losses can be found in the code.

## Testing
Testing was done with the remaining 20% of the data. 

The final losses for each of the stores in testing was:
<pre>
1: 0.02602
2: 0.00502
3: 0.00514
4: 0.00535
5: 0.00395
6: 0.00541
7 :0.0039
8: 0.00446
9: 0.00789
10: 0.00509
11: 0.00617
12: 0.00494
13: 0.00599
14: 0.00668
15: 0.01109
16: 0.01004
17: 0.00825
18: 0.0088
19: 0.00575
20: 0.00918
21: 0.00754
22: 0.11981
23: 0.00312
24: 0.02019
25: 0.00556
26: 0.0085
27: 0.00632
28: 0.00699
29: 0.00731
30: 0.00651
31: 0.00979
32: 0.0277
33: 0.00584
34: 0.01644
35: 0.00453
36: 0.00667
37: 0.00538
38: 0.00621
39: 0.08674
40: 0.00641
41: 0.01217
42: 0.00708
43: 0.01046
44: 0.01223
45: 0.00591
46: 0.0079
47: 0.00514
48: 0.00633
49: 0.00725
50: 0.0063
51: 0.0078
52: 0.09061
53: 0.00669
54: 0.00559
</pre>

A graph of the model prediction vs actual total sale value for each store can be found in the code.

## Ryan Kunkel

My model uses LSTM architecture combined with perceptron networkthat attempts to predict sales for individual product families in each store. I made some adjustments to the model architecture to minimize loss, including modifying the look back size and number of units in the LSTM layer. During model training, I noticed that the model was taking too long to train and that normalizing the data might help in speeding up the process. I am slightly concerned how null / missing data should.


the model seems to overfit some of the stores and not work fine for others. This is an interesting problem Im not sure how to solve yet. perhaps we can incorporate multiple models and combine their outputs in a more sophisticated architecture

otuput:
loss: 68731.2812 - val_loss: 68936.9219
loss: 145094.6719 - val_loss: 148184.5625
loss: 790753.2500 - val_loss: 851369
loss: 100349.7812 - val_loss: 109818.4844
loss: 59552.1133 - val_loss: 72739.2031
loss: 158472.6250 - val_loss: 165084.375
loss: 209442.3750 - val_loss: 213460.8125 
loss: 243276.3594 - val_loss: 254399.26
loss: 229779.5156 - val_loss: 281308.0938
loss: 26464.5879 - val_loss: 37144.785
loss: 237732.5938 - val_loss: 302056.8438
loss: 32679.9277 - val_loss: 56478.0938
loss: 43351.0938 - val_loss: 65663.9141
loss: 32521.3477 - val_loss: 63108.8906
loss: 34407.4844 - val_loss: 57926.2266
loss: 36683.5273 - val_loss: 57021.4219


Testing
Mean Squared Error:  95623.00938605644
Mean Absolute Error: 72.70468859482074


