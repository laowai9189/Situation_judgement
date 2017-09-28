# 中国象棋残局数据集
#### 对应表
棋谱库|特征数据|标签
-----|-----|-----
华山快棋3月|kuai3tezheng.mat|kuai3leibiao.mat
华山快棋4月|kuai4tezheng.mat|kuai4leibiao.mat
华山快棋5月|kuai5tezheng.mat|kuai5leibiao.mat
华山快棋6月|kuai6tezheng.mat|kuai6leibiao.mat
华山快棋7月|kuai7tezheng.mat|kuai7leibiao.mat
华山快棋8月|kuai8tezheng.mat|kuai8leibiao.mat
#### 特征
棋子|存在性|位置信息|基于位置的价值|走法数|机动性|受到的威胁值|受到的保护值|攻击力
--|--|--|--|--|--|--|--|--
黑将|1|2|3|4|5|6|7|8
红帅|9|10|11|12|13|14|15|16
黑车|17|18|19|20|21|22|23|24
黑车|25|26|27|28|29|30|31|32
红车|33|34|35|36|37|38|39|40
#### 标签
所在棋谱最后的对弈结果，红方胜利是1，黑方胜利是2
