## Project for Information Extraction.

To run files, after installing all dependency, run

``` 
python main.py --feature_type <feature_type> --system <system_type>
```

in the root directory.

The argument for feature type can be either `discrete` or `mfcc`.

The argument for system can be either `vanilla` or `contrastive`.

This will run and save the results in the `results` directory. 
pre-ran results are already populated in the `results` directory.

Also, sometimes contrastive model stops at early epoch due to unstableness. Added a patience parameter to training.
With current configuration, if the dev accuracy doesn't improve for 5 epochs, second step of constrastive learning will be started.

Also added train / dev accuracy for fun.


