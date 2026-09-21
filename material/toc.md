# lab 1

Module 1: Unsupervised Foundations & Distance Metrics

1. Supervised vs Unsupervised Learning
2. What Is Clustering?
3. Clustering Does Not Know the "Correct" Groups
4. Similarity and Distance
5. Question 1: Calculate a Distance
6. Why Does Distance Matter in Clustering?
7. Use a Real-World Dataset: Mall Customers
8. Install the Required Libraries
9. Import the Libraries
10. Load the Mall Customers Data
11. Inspect the Dataset
12. Select the Clustering Features
13. Inspect the Shape of the Data
14. Visualize the Customer Data
15. Question 3: Examine the Customer Data

Module 2: Centroid-Based Clustering (K-Means)

15. K-Means Clustering
16. What Does K Mean?
17. The Basic K-Means Algorithm
18. Create a K-Means Model
19. Fit the Model
20. Obtain the Cluster Labels
21. Important: Cluster Numbers Have No Inherent Meaning
22. Question 4: Understand Cluster Labels
23. Visualize the K-Means Clusters
24. What Is a Centroid?  
25. Question 5: Calculate a Centroid
26. Inspect the Centroids Numerically
27. Question 6: Understand cluster_centers_
28. How Does K-Means Decide Which Cluster Gets an Observation?
29. Question 7: Assign an Observation to a Cluster
30. A First Limitation of K-Means
31. Experiment with a Different K
32. Question 8: Changing K

Module 3: Connectivity-Based Clustering (Hierarchical Agglomerative)

33. Hierarchical Clustering
34. K-Means vs Hierarchical Clustering
35. Create an Agglomerative Model
36. Fit the Hierarchical Model
37. Visualize the Hierarchical Clusters
38. Question 9: Compare the Two Methods
39. The Meaning of "Hierarchical"
40. Dendrograms
41. Dendrogram Demonstration
42. Question 10: Read the Dendrogram Conceptually
43. Understanding Cluster Membership

Module 4: Evaluation, Interpretation & Practical Limitations

44. Question 11: Change the Features
45. A Very Important Issue: Feature Scale
46. A Small Example of Scale
47. Question 12: Why Can Scale Matter?
48. First Clustering Experiment with Scaling
49. K-Means Limitations
50. Hierarchical Clustering Limitations 
51. Question 13: Choosing Between K-Means and Hierarchical Clustering
52. Supervised vs Unsupervised Learning
53. Clustering vs Classification
54. A Note About Labels
55. Question 14: Is This Supervised or Unsupervised?
56. Cluster Labels Are Not the Same as Target Labels
57. Question 15: Interpret Cluster Numbers
58. A First Look at Cluster Interpretation
59. Question 16: Interpret Clusters
60. Does Clustering Find "Truth"?
61. Question 17: Is There Always One Correct Clustering?
62. Mini Experiment: Change the Cluster Spread
63. Question 18: Why Does Overlap Matter?
64. Important Insight: Clustering Always Produces a Structure
65. What Happens with Random Data?
66. Question 19: Does K-Means Prove That Groups Exist?
67. What Has Been Learned?
68. K-Means vs Hierarchical Clustering
69. Regression/Classification vs Clustering
70. Where Clustering Is Used
71. What Clustering Does Not Tell Us Automatically
72. Final Reflection
73. Final Knowledge Check

----

# lab 2


Module 1: Diagnostics & Choosing K

1. Start with a Dataset
2. Question 1: Inspect the Data Before Clustering
3. Visualize the Dataset
4. Run K-Means with the Expected Number of Clusters
4. Run K-Means with the Expected Number of Clusters
5. Inspect the Cluster Sizes
6. Question 2: Interpret Cluster Sizes
7. Visualize the Clusters
8. Inertia
9. Why Inertia Alone Is Not Enough
10. Question 3: Why Not Choose the Largest K?
11. The Elbow Method
12. Interpreting the Elbow
13. Question 4: Find the Elbow
14. Clustering Performance : Silhouette Score
15. Calculate Silhouette Scores
16. Question 5: Compare Elbow and Silhouette
17. Why Is K Not Always Objective?

Module 2: Feature Geometry & Scaling

18. Feature Scaling and Clustering
19. Create a Dataset with Different Feature Scales
20. Standardize the Features
21. Question 6: Why Scale Features?
22. Scale a Clustering Dataset
23. A Critical Preprocessing Rule


Module 3: Hierarchical Linkage & Dendrogram Cuts

24. Agglomerative Hierarchical Clustering
25. Linkage
26. Run Agglomerative Clustering
27. Visualize the Result
28. Question 7: Compare the Results
29. Divisive Hierarchical Clustering: A Top-Down Approach
30. Question 10: Compare Agglomerative and Divisive Clustering
31. Understanding the Dendrogram
32. Question 8: Where Could the Hierarchy Be Cut?
33. Try Different Linkage Methods
34. Question 9: Why Does Linkage Matter?

Module 4: Anomaly Detection via Isolation Forest

35. What Is Anomaly Detection?
36. Create Data with Potentially Unusual Transactions
37. Question 10: Identify the Potential Anomalies
38. Isolation Forest
39. Create the Isolation Forest Model
40. Fit and Predict
41. Visualize the Anomalies
42. Question 11: Anomaly vs Cluster
43. Important Caution About Anomalies

Module 5: Association Rule Learning via Apriori

44. A Different Type of Unsupervised Learning
45. Install mlxtend
46. Create a Transaction Dataset
47. Convert Transactions into a Matrix
48. Frequent Itemsets
49. Support
50. Question 12: Calculate Support
51. Generate Association Rules
52. Confidence
53. Question 13: Calculate Confidence
54. Lift
55. Worked Lift Example
56. Question 14: Interpret Lift
57. Association Is Not Causation
58. Question 15: Association or Causation?

Module 6: Comparative Synthesis & Case Scenarios

59. Different Questions, Different Methods
60. Clustering vs Anomaly Detection
61. Clustering vs Association Rules
62. Question 16: Choose the Technique
63. Choosing Between K-Means and Hierarchical Clustering
64. Question 17: Algorithm Selection
65. A More Detailed K-Means Experiment
66. Question 18: Model Selection
67. Cluster Profiling
68. Question 19: Why Profile Clusters?

Part F: Integrated Analysis

69. Complete Unsupervised-Learning Workflow
70. Integrated Scenario
71. Integrated Scenario: Anomaly Detection
72. Integrated Scenario: Association Rules
73. Final Comparison
74. Final Knowledge Check
75. Final Reflection
76. Activity Summary
