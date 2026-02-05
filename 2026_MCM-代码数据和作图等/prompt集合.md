

**摘要**



For Model III，

Through XGBoost modeling and SHAP analysis, we demonstrated that the professional partner is the primary determinant of competition trends. Simultaneously, we quantified the misalignment between judge and audience evaluation standards: the judge system is highly technology-oriented (sensitive to age), while the audience system incorporates more inclusivity and preference for the contestant's background and social attributes.



**下面，请作为论文受，基于上面的工作，撰写论文片段，详细介绍我们建模的过程
可以用的可视化图片：model\_performance\_evaluation.png，shap\_beeswarm\_comparison.png和shap\_importance\_comparison.png**

**先写中文版，引用对应图片**



**好的，请据此翻译成英语，并生成LaTeX代码，最高的标题用\\subsection**



**请采用合适的方式对模型的性能进行评估，并做可视化图，图片的主要配色，按顺序采用：
rank1\_color = (77/255, 103/255, 164/255)    # #4D67A4**

**rank2\_color = (140/255, 141/255, 197/255)  # #8C8DC5**

**rank3\_color = (186/255, 168/255, 210/255)  # #BAA8D2**

**rank4\_color = (237/255, 187/255, 199/255)  # #EDBBC7**

**rank5\_color = (255/255, 204/255, 154/255)  # #FFCC9A**

**rank6\_color = (246/255, 162/255, 126/255)  # #F6A27E**

**rank7\_color = (189/255, 115/255, 106/255)  # #BD736A**

**rank8\_color = (121/255, 77/255, 72/255)    # #794D48**



**好的，下面请对analyze\_impact\_xgboost.py运行得到的图片做一些修改：**

**图片的横向长度都要拉长**

**其他要求：**

**shap\_importance\_comparison.png的柱状图的配色，按照顺序采用：**

**rank1\_color = (77/255, 103/255, 164/255)    # #4D67A4**

**rank3\_color = (186/255, 168/255, 210/255)  # #BAA8D2**

**rank5\_color = (255/255, 204/255, 154/255)  # #FFCC9A**

**rank7\_color = (189/255, 115/255, 106/255)  # #BD736A**

**数字的颜色用rank8\_color = (121/255, 77/255, 72/255)    # #794D48**

**请重新运行程序前两个图：shap\_beeswarm\_comparison.png和shap\_importance\_comparison.png**





**可以做的图：（参考立轩figure 8）34个season每个季度的冠军，最后一week的xx（粉丝投票比例）**



**第二问建模思路笔记**

为了对比百分比制度和排名制度这两种合并方法，我们基于第一问得到的粉丝投票预测结果，对同样的season分别应用百分比制和排名制来进行淘汰，鉴于连续的推演可能会遇到数据确实的问题（比如A本来在weeki被淘汰，那么在weeki+1就没有裁判打分的数据了，如果我们的模型预测他在weeki没有被淘汰，那么要计算他在weeki+1就没有裁判打分的数据了，也就没有办法计算分数），而通过预测值插补或者直接使用最后一周的可用分数的方式都可能引入更大误差，我们以周为单位计算他们的差异，具体来说，我们只在真实的每一周人员内部进行排名，决定谁被淘汰，并将结果和另一制度下的淘汰结果相比较，并且统计淘汰情况不一致的周所占的比例和**最终的排名差异**。

另外，考虑到28-34season的每周淘汰情况会受到裁判对排名最后两位选手评价的影响，我们只选用了前27season进行上述操作。最终得到，在前27season的235个淘汰周中，有69个淘汰周出现分歧，占比29.3%，说明两种融合制度确实对于淘汰选择确实存在明显差异。

为了量化不同赛制的选择是否“ favor fan votes more than the other”，我们统计了每个淘汰周的排名结果和预测的粉丝投票结果之间的差异dis\_Fan\_perentage，和dis\_Fan\_rank并进而求平均值得到dis\_avg\_Fan\_perentage和dis\_avg\_Fan\_rank

为了消除量纲的影响，我们采用同样的方式计算出dis\_avg\_Judge\_perentage，和dis\_avg\_Judge\_rank，并用下面的式子计算得到weight\_perentagehe 和weight\_rank

weight\_j = dis\_avg\_Fan\_j/dis\_avg\_Judge\_j

计算结果显示weight\_perentagehe=









**编写代码：反向求百分比.py**

核心公式：x1'=x1+k\*(0.3\*评委分比例+0.7\*(x1'/(x1'+x2'+..+xj'))，其中j是 x1'对应周的选手数量

x\_i是第i名选手获得粉丝投票的百分比，k是该周被淘汰的人的粉丝投票数，x1'=x1+k\*(评委分比例+x1占x1到x5比例)

其中：

x'，最后一周的x'直接从之前线性规划的结果里面获得

k是x该周的粉丝投票占比，从之前的线性规划的结果里面获取

对每一周求得：该周除了被淘汰人员之外的，其余人的粉丝投票的预测值x\_i

输出文件：反向求百分比.py



线性规划：求出最后一周的所有值，和每一周被淘汰掉的人的值k

调研发现：粉丝投票（人气）的占比更高，比例接近

k的置信区间用蒙特卡洛分布来估计

前面三个特征+伴舞者：placement的平均值

others怎么处理：还是按others来算





**逆马尔科夫**

**第28季开始引入倒一倒二（基于总分数）的筛选，**

**评委筛选时选择：**

**特征影响需要重新评估**







**淘汰结果对比**

请利用刚才的数据，和第几week淘汰.csv中的数据进行对比，若淘汰的week匹配匹配，将对应格改成correct，反之，改成ERROR，输出文件：百分制淘汰-和真实情况对比.csv



**第一问**

我的核心关切是，能不能用已知的数据，求出每一周的粉丝投票估计，并且能够用一个指标来衡量预测的确定行， 请分析你提出方案针对上述需求的可行性（需要输入是什么，核心公式是什么，预测出每周粉丝投票的结果如何（具体值还是区间），确定性怎么衡量）





**箱式图others**

请在"cur代码"文件夹内实现代码：1\_dim\_characterastic.py ,基于文件 C\_origin.csv，做三个二维的箱式图，中间部分选择75%区间，y轴都是最终的排名placement，x轴分别考虑celebrity\_industry，celebrity\_homecountry/region，celebrity\_age\_during\_season这三种，最终在"cur代码"文件夹下输出三个箱式统计图片 x.png

注意：industry正常取所有的值，单我不希望age和homestate横轴是所有可能的取值，需要进行归类：其中age按照每10岁划分一个区间，homestate按照大洲来进行划分

注意，在编写代码的时候，要明确定义几个变量，方便我后续修改值：

选取season区间的首位，season\_start和season\_final



**rank1\_color = (77/255, 103/255, 164/255)    # #4D67A4**

**rank3\_color = (186/255, 168/255, 210/255)  # #BAA8D2**

**rank5\_color = (255/255, 204/255, 154/255)  # #FFCC9A**

**rank7\_color = (189/255, 115/255, 106/255)  # #BD736A**



**配色1357**

**rank1\_color = (77/255, 103/255, 164/255)    # #4D67A4**

**rank2\_color = (140/255, 141/255, 197/255)  # #8C8DC5**

**rank3\_color = (186/255, 168/255, 210/255)  # #BAA8D2**

**rank4\_color = (237/255, 187/255, 199/255)  # #EDBBC7**

**rank5\_color = (255/255, 204/255, 154/255)  # #FFCC9A**

**rank6\_color = (246/255, 162/255, 126/255)  # #F6A27E**

**rank7\_color = (189/255, 115/255, 106/255)  # #BD736A**

**rank8\_color = (121/255, 77/255, 72/255)    # #794D48**



**二维图片【靠前是绿色】**

**请在"cur代码"文件夹内实现代码：2\_dim\_characterastic.py ,基于文件 C\_origin.csv，做三个二维的可视化图，xy轴分别考虑celebrity\_industry，celebrity\_homecountry/region，celebrity\_age\_during\_season的三类两两组合，图中的每个坐标点用不同颜色区分，其中颜色由排名placement决定，排名越靠前，越接近薄荷绿色，排名越靠后，越接近浅紫色。**

**最终在"cur代码"文件夹下输出三个统计图片 x\_y.png**

**注意，在编写代码的时候，要明确定义几个变量，方便我后续修改值：**

**排名第一的样本点的颜色rank1\_color**

**排名最后的样本点的颜色rankl\_color(二者均用RGB表示)**

**选取season区间的首位，season\_start和season\_final**





**百分比【Q1预测百分比.py】**

**请在cur代码文件夹内**实现代码：Q1百分比.py，基于百分比制-每周平均占比\&排名.csv中的数据

利用下面这几列作为输入：

评委打分百分比($J\_i$)：来自数据集中的week\_i\_judge\_percentages

评委打分排名：week\_i\_judge\_rank

最终名次（Placement）：来自数据集中的 placement

用代码实现：

连续数学规则是：$\\text{总分} = \\text{评委百分比} + \\text{粉丝百分比}$，总分最高的获胜 。如何解决： 使用线性规划（LP）。如果选手 A 排在选手 B 前面，那么 A 的总分必须大于 B 的总分。我们将这个逻辑写成数学不等式：$(p^J\_A + p^F\_A) \\ge (p^J\_B + p^F\_B) + \\epsilon$。通过解这个不等式组，我们能得到每个人粉丝百分比 ($p^F\_i$) 的最小值和最大值。

最终，通过点估计（最大熵）来选取输出结果： 在满足排名的所有可能投票分布中，选一个“最不极端”的（即熵最大的），作为最终的预测结果，得到每个celebrity 的粉丝得分的区间估计，输出csv文件：百分比-区间估计.csv【

包含列：

（前面的列和原数据保持不变）

celebrity

ballroom\_partner

celebrity\_industry

celebrity\_homestate

celebrity\_homecountry/region

celebrity\_age\_during\_season

season

placement

（后面的数据根据加工结果填写）

第i周的预测百分比min

第i周的预测百分比max

第i周的derta（用max-min计算得到）

第i周的点估计结果

】

】



请基于Q1百分比.py中的数据，利用下面几列作图：
预测百分比min

预测百分比max

点估计结果

week\_i\_judge\_percentages



？这个区间预测是只能预测最后一week的区间是吗？



我要确认一个点，“排名合并制 (Season 1–2, 28–34) —— 离散逻辑中的映射： 排名只是 1, 2, 3，为了变成具体的票数，引入了“指数间隔”假设，认为第 1 名比第 2 名领先的程度符合某种统计分布。”是怎么进行映射的

B. 排名合并制 (Season 1–2, 28–34) —— 离散逻辑规则是：$\\text{总排位} = \\text{评委排名} + \\text{粉丝排名}$，总排位数字越小越好 。如何解决： 使用穷举法（Permutation Search）。因为决赛选手很少（通常 3-5 人），模型会尝试所有可能的粉丝排名组合（如：A 是粉丝第 1，B 是粉丝第 2...）。保留那些计算后能产生与真实名次一致的组合。映射： 排名只是 1, 2, 3，为了变成具体的票数，引入了“指数间隔”假设，认为第 1 名比第 2 名领先的程度符合某种统计分布。



**【第几week淘汰.csv】**

请基于C\_origin.csv进行数据处理,编写代码实现：

统计每个celebrity被eliminate的周（如果i周后面几周，judge的打分都为NA，那么认为他在i周结束后出局）

输出csv文件：第几week淘汰.csv【包含列：

（前面的列和原数据保持不变）

celebrity

ballroom\_partner

celebrity\_industry

celebrity\_homestate

celebrity\_homecountry/region

celebrity\_age\_during\_season

season

placement

（后面的数据根据加工结果填写）

week\_i\_fail（其中i为对应的周）

】



**排名制【排名制-每周平均分数(十分制)\&排名.csv】**

请基于C\_origin.csv进行数据处理,编写代码实现：

对于season = 1,2 或者season  >= 28：

对每个celebrity，计算每一周的平均分数（注意，分数都应该采用十分制，为小数的请自动扩充成10分制），填到下面的week\_i\_judge\_avg\_score列里面，并根据分数计算每个season内该选手在该周的排名，填充到week\_i\_judge\_rank列里面，输出csv文件：每周平均分数(十分制)\&排名.csv【包含列：

（前面的列和原数据保持不变）

celebrity

ballroom\_partner

celebrity\_industry

celebrity\_homestate

celebrity\_homecountry/region

celebrity\_age\_during\_season

season

placement

（后面的数据根据加工结果填写）

week\_i\_judge\_avg\_score

week\_i\_judge\_rank（其中i为对应的周）

】



**百分比制度【百分比制-每周平均占比\&排名.csv】**

请基于C\_origin.csv进行数据处理,编写代码实现：

对season>=3 ^ season<=27:

对每个celebrity，计算其在该周的选手的法官百分比

公式为法官百分比 = (该选手当周总分) / (该season中所有参赛选手当周法官总分之和)，填到下面的week\_i\_judge\_percentages列里面，并根据占比计算每个season内该选手在该周的排名（占比大的排名靠前）填充到week\_i\_judge\_rank列里面，输出csv文件：百分比制-每周平均占比\&排名.csv【包含列：

（前面的列和原数据保持不变）

celebrity

ballroom\_partner

celebrity\_industry

celebrity\_homestate

celebrity\_homecountry/region

celebrity\_age\_during\_season

season

placement

（后面的数据根据加工结果填写）

week\_i\_judge\_percentages

week\_i\_judge\_rank（其中i为对应的周）

】

