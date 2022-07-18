# image-privacy

This is a Java project that contains the experimental testbed that we used in the following publication:

E. Spyromitros-Xioufis, S. Papadopoulos, A. Popescu, Y. Kompatsiaris, "<a href="http://users.auth.gr/espyromi/publications/papers/ICMR2016.pdf">Personalized Privacy-aware Image Classification</a>", Proc. International Conference on Multimedia Retrieval (ICMR), New York, USA, June 6-9, 2016.

All the experimental results of the paper can be replicated by simply executing the main method of the ExperimentsRunner class (after setting the datasetFolder variable to point at the location where the datasets reside).

The datasets can be found at: <a href="https://drive.google.com/file/d/1j-9e1EuOuqiikXf5-y4CQiMB1P2WYRlM/view">the following link</a> 
and are divided in two subfolders, '/youralert' and '/picalert', each one containing the datasets that we created out of the images and ground truth of the corresponding image privacy collection (YourAlert/PicAlert). For both datasets we have extracted 'vlad', 'cnn', and 'semfeat' features (as described in our paper) and, additionally, 'edch' and 'bow' features have been kindly provided for PicAlert from the <a href="http://l3s.de/picalert/">PicAlert team</a>. Thus, we have composed 3 YourAlert and 5 PicAlert datasets.
For all datasets we use <a href="http://www.cs.waikato.ac.nz/~ml/weka/">Weka</a>'s <a href="http://www.cs.waikato.ac.nz/ml/weka/arff.html">sparse ARFF format</a> with a header that in all cases looks like:
<pre>
@ATTRIBUTE id String
@ATTRIBUTE user {u11, u10, u13, u12, u15, u14, u17, u16, u1, u19, u2, u18, u3, u4, u5, u6, u7, u8, u9, u20, u22, u21, u24, u23, u26, u25, u27}
@ATTRIBUTE source {picalert, youralert}
@ATTRIBUTE feat_1 numeric
@ATTRIBUTE feat_2 numeric
...
@ATTRIBUTE feat_K numeric
@ATTRIBUTE private {0,1}
</pre>
where the 'id' attribute contains the photo id, the 'user' attribute contains the user id (always missing for PicAlert), the 'source' attribute has the value 'picalert' for PicAlert and 'youralert' for 'YourAlert', 'feat\_1',...,'feat\_K' are the visual features and, finally, 'private' is the class attribute that is '0' for public and '1' for private images.

