	//区域筛选
	size_t min;
	double a1=0, a2=9999;
	double total_x=0, total_y=0;
	unsigned char i_size;
	if(matches.size() < 5)
		i_size = matches.size();
	else
		i_size = 5;
	for (unsigned char i=0; i<i_size; i++ )
	{
		for(size_t j=0; j<matches.size(); j++)
		{
			if ( matches[j].distance < a2 && matches[j].distance > a1)
			{
				a2 = matches[j].distance;
				min = j;
			}
		}
		total_x += keypointsB[matches[min].trainIdx].pt.x;
		total_y += keypointsB[matches[min].trainIdx].pt.y;
		a1 = a2;
		a2 = 9999;
	}
	//几何中心坐标
	total_x = total_x / 5;
	total_y = total_y / 5;




	/*old goodkeypoints way
	match_minDis = 9999;
	goodMatches.clear();
	goodkeypointsA.clear();
	goodkeypointsB.clear();
	for ( size_t i=0; i<matches.size(); i++ )
	{
		if ( matches[i].distance < match_minDis )
			match_minDis = matches[i].distance;
	}

	for ( size_t i=0; i<matches.size(); i++ )
	{
		if (matches[i].distance < 3*match_minDis)
		{
			goodMatches.push_back(matches[i]);
			goodkeypointsA.push_back(keypointsA[matches[i].queryIdx].pt);
			goodkeypointsB.push_back(keypointsB[matches[i].trainIdx].pt);
		}
	}
	*/



	match_minDis = 9999;
	//int min;
	for ( size_t i=0; i<pre_matches.size(); i++ )
	{
		if ( pre_matches[i].distance < match_minDis )
		{
			match_minDis = pre_matches[i].distance;
			min = i;
		}
	}