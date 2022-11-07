# wave-energy-converters

Wave energy converters (WECs), in the form of fully submerged buoys, are used to extract energy from waves. As waves propagate through the surrounding water, energy extraction is enabled by changes in the tension on the tethers which connecteach WEC to the sea floor. To maximize the energy generation, WECs must be able to exploit the wave conditions during a given period while also maximizing the constructive wave interference and minimizing the destructive interference between the WECs. This sets the stage for an optimization problem, wherein the goal is to maximize the total energy generation potential of an arrangement of WECs.

Data Set: <a href = "https://archive.ics.uci.edu/ml/datasets/Wave+Energy+Converters"><i> Wave Energy Converters Data Set </i></a>
The data set consists of 288,000 “real wave scenarios”. from four different coastal cities in Australia. A sample describes a configuration of 16 wave energy converting buoys, and their power outputs. The configuration of the buoys exists in a size-constrained environment. A sample has 49 real numbers. The first 32 numbers are the <i>x</i> and <i>y</i> coordinates of the WECs. The next 16 numbers are the power outputs for each of the 16 WECs. Finally, the last number is the sum of power production from all the buoys in the configuration. This dataset is not trivial because WECs can destructively and constructively
interfere with each other's power output according to complex hydrodynamics that depend on the specific characteristics of the scenario. This data was generated in a “Wave Energy Converter (WEC) Array Simulator” written by Nataliia Sergiienko.

We propose to perform a regression on this data set to create a model. Using the
data, we can try to find a function f:R^{32}⟶R^{17} that will take the WEC positions (16 <i>x</i> coordinates and 16 <i>y</i> coordinates) as an input, and output a prediction of the total power output as well as the power of each individual buoy. This could be used for planning of new WEC placements. We plan to test a couple of different basic regression techniques (linear) and neural network architectures (non-linear) for the task and evaluate the accuracy of the models by their total least-square error on a test set.

Each WEC has a radius of 5m and is submerged to a depth of 3m below sea level. The angle of the tether that connects each WEC to the sea floor, which is another 27m deeper, is constant at 55^{o}. WEC_{i} has position (x_{i}, y_{i}) which is constrained to a square area of 20,000 m^{2} per WEC or 320,000 m^{2} for the entire array of WECs. Safety considerations impose an additional lower limit of at least 50m spacing (Euclidean distance) between each
WEC.

### Resources
On the UCI Machine Learning Repository there are 3 papers written on this subject that could be of interest to cite for background information or ideas for further work:
1. L. D. Mann, A. R. Burns, , and M. E. Ottaviano. 2007. CETO, a carbon free wave power energy provider of the future. In the 7th European Wave and Tidal Energy Conference (EWTEC).
2. Neshat, M., Alexander, B., Wagner, M., & Xia, Y. (2018, July). A detailed comparison of meta-heuristic methods for optimising wave energy converter placements. In Proceedings of the Genetic and Evolutionary Computation Conference (pp. 1318-1325). ACM.
3. Neshat, M., Alexander, B., Sergiienko, N., & Wagner, M. (2019). A new insight into the Position Optimization of Wave Energy Converters by a Hybrid Local Search.

References:
Dua, D. and Graff, C. (2019). UCI Machine Learning Repository
[http://archive.ics.uci.edu/ml]. Irvine, CA: University of California, School of
Information and Computer Science.
