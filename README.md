## Towards a General Theory of Infinite-Width Limits of Neural Classifiers

This is the code to reproduce results of the paper *"Towards a General Theory of Infinite-Width Limits of Neural Classifiers"* 
submitted to ICML'20.

### Instructions:

First, perform computations:
* Main body:
  * Figure 1:
    * Left and center:
    ```
    $ python fcnet_width_dependence.py --device=<cuda:0 or cpu> --num_hidden=1 --optimizer=sgd --lr=0.02 --num_epochs=50 --dataset=cifar2_binary --train_size=1000 --batch_size=1000 --activation=lrelu --num_seeds=5
    ```
    * Right:
    ```
    $ python epoch_dependence.py --device=<cuda:0 or cpu> --num_hidden=1 --optimizer=sgd --lr=0.02 --num_epochs=50 --dataset=cifar2_binary --train_size=1000 --batch_size=1000 --activation=lrelu --num_seeds=5  
    ```
  * Figure 2:
    * Left:
    ```
    $ python fcnet_width_dependence.py --device=<cuda:0 or cpu> --num_hidden=2 --optimizer=sgd --lr=0.02 --num_epochs=50 --dataset=cifar2_binary --train_size=1000 --batch_size=1000 --activation=lrelu --num_seeds=5
    ```
    * Center:
    ```
    $ python fcnet_width_dependence.py --device=<cuda:0 or cpu> --num_hidden=3 --optimizer=sgd --lr=0.02 --num_epochs=50 --dataset=cifar2_binary --train_size=1000 --batch_size=1000 --activation=lrelu --num_seeds=5
    ```
    * Right:
    ```
    $ python fcnet_width_dependence.py --device=<cuda:0 or cpu> --num_hidden=3 --optimizer=rmsprop --lr=0.0002 --num_epochs=50 --dataset=cifar2_binary --train_size=1000 --batch_size=1000 --activation=lrelu --num_seeds=5
    ```
* Supplementary material:
  * Figure 1:
    * All plots:
    ```
    $ python fcnet_width_dependence.py --device=<cuda:0 or cpu> --num_hidden=1 --optimizer=sgd --lr=0.02 --num_epochs=50 --dataset=cifar2_binary --train_size=1000 --batch_size=1000 --activation=lrelu --num_seeds=5
    ```
  * Figure 2:
    * All plots:
    ```
    $ python fcnet_width_dependence.py --device=<cuda:0 or cpu> --num_hidden=1 --optimizer=sgd --lr=0.02 --num_epochs=50 --dataset=cifar2_binary --train_size=1000 --batch_size=1000 --activation=lrelu --num_seeds=5
    ```
  * Figure 3:
    * Top row, left and right:
    ```
    $ python fcnet_width_dependence.py --device=<cuda:0 or cpu> --num_hidden=1 --optimizer=sgd --lr=0.02 --num_epochs=50 --dataset=cifar2_binary --train_size=1000 --batch_size=1000 --activation=lrelu --num_seeds=5
    ```
    * Top row, center:
    ```
    $ python epoch_dependence.py --device=<cuda:0 or cpu> --num_hidden=1 --optimizer=sgd --lr=0.02 --num_epochs=50 --dataset=cifar2_binary --train_size=1000 --batch_size=1000 --activation=lrelu --num_seeds=5  
    ```
    * Bottom row, left and right:
    ```
    $ python fcnet_width_dependence.py --device=<cuda:0 or cpu> --num_hidden=1 --optimizer=sgd --lr=0.0002 --num_epochs=50 --dataset=cifar2_binary --train_size=1000 --batch_size=1000 --activation=lrelu --num_seeds=50
    ```
    * Bottom row, center:
    ```
    $ python epoch_dependence.py --device=<cuda:0 or cpu> --num_hidden=1 --optimizer=sgd --lr=0.0002 --num_epochs=50 --dataset=cifar2_binary --train_size=1000 --batch_size=1000 --activation=lrelu --num_seeds=5  
    ```
  * Figure 4:
    * Top row:
    ```
    $ python fcnet_width_dependence.py --device=<cuda:0 or cpu> --num_hidden=1 --optimizer=sgd --lr=0.02 --num_epochs=50 --dataset=cifar2_binary --train_size=1000 --batch_size=1000 --activation=lrelu --num_seeds=5
    ```
    * Second row:
    ```
    $ python fcnet_width_dependence.py --device=<cuda:0 or cpu> --num_hidden=1 --optimizer=sgd --lr=0.02 --num_epochs=5 --dataset=cifar2_binary --train_size=1000 --batch_size=100 --activation=lrelu --num_seeds=5
    ```
    * Third row:
    ```
    $ python fcnet_width_dependence.py --device=<cuda:0 or cpu> --num_hidden=1 --optimizer=sgd --lr=0.02 --num_epochs=5 --dataset=cifar2_binary --train_size=10000 --batch_size=1000 --activation=lrelu --num_seeds=5
    ```
    * Bottom row:
    ```
    $ python fcnet_width_dependence.py --device=<cuda:0 or cpu> --num_hidden=1 --optimizer=sgd --lr=0.02 --num_epochs=500 --dataset=cifar2_binary --train_size=1000 --batch_size=1000 --activation=lrelu --num_seeds=5
    ```
  * Figure 5:
    * Top row:
    ```
    $ python fcnet_width_dependence.py --device=<cuda:0 or cpu> --num_hidden=1 --optimizer=sgd --lr=0.02 --num_epochs=50 --dataset=cifar2_binary --train_size=1000 --batch_size=1000 --activation=lrelu --num_seeds=5
    ```
    * Middle row:
    ```
    $ python fcnet_width_dependence.py --device=<cuda:0 or cpu> --num_hidden=1 --optimizer=sgd --lr=0.02 --num_epochs=50 --dataset=mnist --train_size=1000 --batch_size=1000 --activation=lrelu --num_seeds=5
    ```
    * Bottom row:
    ```
    $ python fcnet_width_dependence.py --device=<cuda:0 or cpu> --num_hidden=1 --optimizer=sgd --lr=0.02 --num_epochs=10 --dataset=mnist --train_size=60000 --batch_size=100 --activation=lrelu --num_seeds=5
    ```
   
After performing computations, make plots by running cells in notebooks: [width_dependence_plots.ipynb](/width_dependence_plots.ipynb) and [epoch_dependence_plots.ipynb](/epoch_dependence_plots.ipynb).
