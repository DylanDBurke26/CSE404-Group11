# CSE 404 Project Group 11


## Authors

- [Austin LeBlanc](https://github.com/austinleblanc)
- [Dylan Burke](https://github.com/DylanDBurke26)
- [Ryan Kunkel](https://github.com/rykunk21)


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
### Person 2