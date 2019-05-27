Search.setIndex({docnames:["index","modules","pyqstrat"],envversion:{"sphinx.domains.c":1,"sphinx.domains.changeset":1,"sphinx.domains.cpp":1,"sphinx.domains.javascript":1,"sphinx.domains.math":2,"sphinx.domains.python":1,"sphinx.domains.rst":1,"sphinx.domains.std":1,"sphinx.ext.viewcode":1,sphinx:56},filenames:["index.rst","modules.rst","pyqstrat.rst"],objects:{"":{pyqstrat:[2,0,0,"-"]},"pyqstrat.account":{Account:[2,1,1,""],ContractPNL:[2,1,1,""],calc_trade_pnl:[2,3,1,""],find_index_before:[2,3,1,""],find_last_non_nan_index:[2,3,1,""],leading_nan_to_zero:[2,3,1,""],test_account:[2,3,1,""]},"pyqstrat.account.Account":{__init__:[2,2,1,""],add_trades:[2,2,1,""],calc:[2,2,1,""],df_account_pnl:[2,2,1,""],df_pnl:[2,2,1,""],df_trades:[2,2,1,""],equity:[2,2,1,""],position:[2,2,1,""],positions:[2,2,1,""],symbols:[2,2,1,""],trades:[2,2,1,""]},"pyqstrat.account.ContractPNL":{calc_net_pnl:[2,2,1,""],df:[2,2,1,""],net_pnl:[2,2,1,""],pnl:[2,2,1,""],position:[2,2,1,""]},"pyqstrat.evaluator":{Evaluator:[2,1,1,""],compute_amean:[2,3,1,""],compute_annual_returns:[2,3,1,""],compute_bucketed_returns:[2,3,1,""],compute_calmar:[2,3,1,""],compute_dates_3yr:[2,3,1,""],compute_equity:[2,3,1,""],compute_gmean:[2,3,1,""],compute_mar:[2,3,1,""],compute_maxdd_date:[2,3,1,""],compute_maxdd_date_3yr:[2,3,1,""],compute_maxdd_pct:[2,3,1,""],compute_maxdd_pct_3yr:[2,3,1,""],compute_maxdd_start:[2,3,1,""],compute_maxdd_start_3yr:[2,3,1,""],compute_num_periods:[2,3,1,""],compute_periods_per_year:[2,3,1,""],compute_return_metrics:[2,3,1,""],compute_returns_3yr:[2,3,1,""],compute_rolling_dd:[2,3,1,""],compute_rolling_dd_3yr:[2,3,1,""],compute_sharpe:[2,3,1,""],compute_sortino:[2,3,1,""],compute_std:[2,3,1,""],display_return_metrics:[2,3,1,""],handle_non_finite_returns:[2,3,1,""],plot_return_metrics:[2,3,1,""],test_evaluator:[2,3,1,""]},"pyqstrat.evaluator.Evaluator":{__init__:[2,2,1,""],add_metric:[2,2,1,""],compute:[2,2,1,""],compute_metric:[2,2,1,""],metric:[2,2,1,""],metrics:[2,2,1,""]},"pyqstrat.holiday_calendars":{Calendar:[2,1,1,""],read_holidays:[2,3,1,""]},"pyqstrat.holiday_calendars.Calendar":{EUREX:[2,4,1,""],NYSE:[2,4,1,""],__init__:[2,2,1,""],add_calendar:[2,2,1,""],add_trading_days:[2,2,1,""],get_calendar:[2,2,1,""],get_trading_days:[2,2,1,""],is_trading_day:[2,2,1,""],num_trading_days:[2,2,1,""]},"pyqstrat.marketdata_processor":{PathFileNameProvider:[2,1,1,""],SingleDirectoryFileNameMapper:[2,1,1,""],TextHeaderParser:[2,1,1,""],base_date_filename_mapper:[2,3,1,""],create_text_file_processor:[2,3,1,""],get_field_indices:[2,3,1,""],process_marketdata:[2,3,1,""],process_marketdata_file:[2,3,1,""],text_file_record_generator:[2,3,1,""]},"pyqstrat.marketdata_processor.PathFileNameProvider":{__call__:[2,2,1,""],__init__:[2,2,1,""]},"pyqstrat.marketdata_processor.SingleDirectoryFileNameMapper":{__call__:[2,2,1,""],__init__:[2,2,1,""]},"pyqstrat.marketdata_processor.TextHeaderParser":{__call__:[2,2,1,""],__init__:[2,2,1,""]},"pyqstrat.optimize":{Experiment:[2,1,1,""],Optimizer:[2,1,1,""],test_optimize:[2,3,1,""]},"pyqstrat.optimize.Experiment":{cost:[2,4,1,""],other_costs:[2,4,1,""],suggestion:[2,4,1,""],valid:[2,2,1,""]},"pyqstrat.optimize.Optimizer":{__init__:[2,2,1,""],df_experiments:[2,2,1,""],experiment_list:[2,2,1,""],plot_2d:[2,2,1,""],plot_3d:[2,2,1,""],run:[2,2,1,""]},"pyqstrat.orders":{LimitOrder:[2,1,1,""],MarketOrder:[2,1,1,""],RollOrder:[2,1,1,""],StopLimitOrder:[2,1,1,""]},"pyqstrat.orders.LimitOrder":{__init__:[2,2,1,""]},"pyqstrat.orders.MarketOrder":{__init__:[2,2,1,""]},"pyqstrat.orders.RollOrder":{__init__:[2,2,1,""]},"pyqstrat.orders.StopLimitOrder":{__init__:[2,2,1,""]},"pyqstrat.plot":{BucketedValues:[2,1,1,""],DateFormatter:[2,1,1,""],DateLine:[2,1,1,""],HorizontalLine:[2,1,1,""],Plot:[2,1,1,""],Subplot:[2,1,1,""],TimeSeries:[2,1,1,""],TradeBarSeries:[2,1,1,""],TradeSet:[2,1,1,""],VerticalLine:[2,1,1,""],XYData:[2,1,1,""],XYZData:[2,1,1,""],draw_3d_plot:[2,3,1,""],draw_boxplot:[2,3,1,""],draw_candlestick:[2,3,1,""],draw_date_line:[2,3,1,""],draw_horizontal_line:[2,3,1,""],draw_poly:[2,3,1,""],draw_vertical_line:[2,3,1,""],get_date_formatter:[2,3,1,""],test_plot:[2,3,1,""],trade_sets_by_reason_code:[2,3,1,""]},"pyqstrat.plot.BucketedValues":{__init__:[2,2,1,""]},"pyqstrat.plot.Plot":{__init__:[2,2,1,""],draw:[2,2,1,""]},"pyqstrat.plot.Subplot":{__init__:[2,2,1,""],get_all_timestamps:[2,2,1,""]},"pyqstrat.plot.TimeSeries":{__init__:[2,2,1,""],reindex:[2,2,1,""]},"pyqstrat.plot.TradeBarSeries":{__init__:[2,2,1,""],df:[2,2,1,""],reindex:[2,2,1,""]},"pyqstrat.plot.TradeSet":{__init__:[2,2,1,""],reindex:[2,2,1,""]},"pyqstrat.plot.XYData":{__init__:[2,2,1,""]},"pyqstrat.plot.XYZData":{__init__:[2,2,1,""]},"pyqstrat.portfolio":{Portfolio:[2,1,1,""]},"pyqstrat.portfolio.Portfolio":{__init__:[2,2,1,""],add_strategy:[2,2,1,""],df_returns:[2,2,1,""],evaluate_returns:[2,2,1,""],plot:[2,2,1,""],run:[2,2,1,""],run_indicators:[2,2,1,""],run_rules:[2,2,1,""],run_signals:[2,2,1,""]},"pyqstrat.pq_types":{Contract:[2,1,1,""],ContractGroup:[2,1,1,""],OrderStatus:[2,1,1,""],Trade:[2,1,1,""]},"pyqstrat.pq_types.Contract":{clear:[2,5,1,""],create:[2,5,1,""]},"pyqstrat.pq_types.ContractGroup":{add_contract:[2,2,1,""],clear:[2,5,1,""],create:[2,5,1,""],get_contract:[2,2,1,""]},"pyqstrat.pq_types.OrderStatus":{FILLED:[2,4,1,""],OPEN:[2,4,1,""]},"pyqstrat.pq_types.Trade":{__init__:[2,2,1,""]},"pyqstrat.pq_utils":{ReasonCode:[2,1,1,""],date_2_num:[2,3,1,""],day_of_week_num:[2,3,1,""],decode_future_code:[2,3,1,""],get_empty_np_value:[2,3,1,""],get_fut_code:[2,3,1,""],get_temp_dir:[2,3,1,""],has_display:[2,3,1,""],infer_compression:[2,3,1,""],infer_frequency:[2,3,1,""],is_newer:[2,3,1,""],linear_interpolate:[2,3,1,""],millis_since_epoch:[2,3,1,""],monotonically_increasing:[2,3,1,""],nan_to_zero:[2,3,1,""],np_find_closest:[2,3,1,""],np_get_index:[2,3,1,""],percentile_of_score:[2,3,1,""],resample_trade_bars:[2,3,1,""],resample_ts:[2,3,1,""],resample_vwap:[2,3,1,""],series_to_array:[2,3,1,""],set_defaults:[2,3,1,""],shift_np:[2,3,1,""],str2date:[2,3,1,""],strtup2date:[2,3,1,""],to_csv:[2,3,1,""],touch:[2,3,1,""],zero_to_nan:[2,3,1,""]},"pyqstrat.pq_utils.ReasonCode":{BACKTEST_END:[2,4,1,""],ENTER_LONG:[2,4,1,""],ENTER_SHORT:[2,4,1,""],EXIT_LONG:[2,4,1,""],EXIT_SHORT:[2,4,1,""],MARKER_PROPERTIES:[2,4,1,""],NONE:[2,4,1,""],ROLL_FUTURE:[2,4,1,""]},"pyqstrat.pyqstrat_cpp":{Aggregator:[2,1,1,""],AllOpenInterestAggregator:[2,1,1,""],AllOtherAggregator:[2,1,1,""],AllQuoteAggregator:[2,1,1,""],AllQuotePairAggregator:[2,1,1,""],AllTradeAggregator:[2,1,1,""],ArrowWriter:[2,1,1,""],ArrowWriterCreator:[2,1,1,""],BadLineHandler:[2,1,1,""],CheckFields:[2,1,1,""],FileProcessor:[2,1,1,""],FixedWidthTimeParser:[2,1,1,""],FormatTimestampParser:[2,1,1,""],IsFieldInList:[2,1,1,""],LineFilter:[2,1,1,""],MissingDataHandler:[2,1,1,""],OpenInterestRecord:[2,1,1,""],OtherRecord:[2,1,1,""],PriceQtyMissingDataHandler:[2,1,1,""],PrintBadLineHandler:[2,1,1,""],QuoteRecord:[2,1,1,""],QuoteTOBAggregator:[2,1,1,""],Record:[2,1,1,""],RecordFieldParser:[2,1,1,""],RecordFilter:[2,1,1,""],RecordGenerator:[2,1,1,""],RecordParser:[2,1,1,""],RegExLineFilter:[2,1,1,""],Schema:[2,1,1,""],SubStringLineFilter:[2,1,1,""],TextFileDecompressor:[2,1,1,""],TextFileProcessor:[2,1,1,""],TextOpenInterestParser:[2,1,1,""],TextOtherParser:[2,1,1,""],TextQuotePairParser:[2,1,1,""],TextQuoteParser:[2,1,1,""],TextRecordParser:[2,1,1,""],TextTradeParser:[2,1,1,""],TimestampParser:[2,1,1,""],TradeBarAggregator:[2,1,1,""],TradeRecord:[2,1,1,""],Writer:[2,1,1,""],WriterCreator:[2,1,1,""],black_scholes_price:[2,3,1,""],cdf:[2,3,1,""],d1:[2,3,1,""],d2:[2,3,1,""],delta:[2,3,1,""],gamma:[2,3,1,""],implied_vol:[2,3,1,""],ostream_redirect:[2,1,1,""],rho:[2,3,1,""],theta:[2,3,1,""],vega:[2,3,1,""]},"pyqstrat.pyqstrat_cpp.Aggregator":{__call__:[2,2,1,""],__init__:[2,2,1,""]},"pyqstrat.pyqstrat_cpp.AllOpenInterestAggregator":{__call__:[2,2,1,""],__init__:[2,2,1,""]},"pyqstrat.pyqstrat_cpp.AllOtherAggregator":{__call__:[2,2,1,""],__init__:[2,2,1,""]},"pyqstrat.pyqstrat_cpp.AllQuoteAggregator":{__call__:[2,2,1,""],__init__:[2,2,1,""]},"pyqstrat.pyqstrat_cpp.AllQuotePairAggregator":{__call__:[2,2,1,""],__init__:[2,2,1,""]},"pyqstrat.pyqstrat_cpp.AllTradeAggregator":{__call__:[2,2,1,""],__init__:[2,2,1,""]},"pyqstrat.pyqstrat_cpp.ArrowWriter":{__init__:[2,2,1,""],add_record:[2,2,1,""],close:[2,2,1,""],write_batch:[2,2,1,""]},"pyqstrat.pyqstrat_cpp.ArrowWriterCreator":{__call__:[2,2,1,""],__init__:[2,2,1,""]},"pyqstrat.pyqstrat_cpp.BadLineHandler":{__call__:[2,2,1,""],__init__:[2,2,1,""]},"pyqstrat.pyqstrat_cpp.CheckFields":{__call__:[2,2,1,""],__init__:[2,2,1,""]},"pyqstrat.pyqstrat_cpp.FileProcessor":{__call__:[2,2,1,""],__init__:[2,2,1,""]},"pyqstrat.pyqstrat_cpp.FixedWidthTimeParser":{__call__:[2,2,1,""],__init__:[2,2,1,""]},"pyqstrat.pyqstrat_cpp.FormatTimestampParser":{__call__:[2,2,1,""],__init__:[2,2,1,""]},"pyqstrat.pyqstrat_cpp.IsFieldInList":{__call__:[2,2,1,""],__init__:[2,2,1,""]},"pyqstrat.pyqstrat_cpp.LineFilter":{__call__:[2,2,1,""],__init__:[2,2,1,""]},"pyqstrat.pyqstrat_cpp.MissingDataHandler":{__call__:[2,2,1,""],__init__:[2,2,1,""]},"pyqstrat.pyqstrat_cpp.OpenInterestRecord":{__init__:[2,2,1,""],id:[2,4,1,""],metadata:[2,4,1,""],qty:[2,4,1,""],timestamp:[2,4,1,""]},"pyqstrat.pyqstrat_cpp.OtherRecord":{__init__:[2,2,1,""],id:[2,4,1,""],metadata:[2,4,1,""],timestamp:[2,4,1,""]},"pyqstrat.pyqstrat_cpp.PriceQtyMissingDataHandler":{__call__:[2,2,1,""],__init__:[2,2,1,""]},"pyqstrat.pyqstrat_cpp.PrintBadLineHandler":{__call__:[2,2,1,""],__init__:[2,2,1,""]},"pyqstrat.pyqstrat_cpp.QuoteRecord":{__init__:[2,2,1,""],bid:[2,4,1,""],id:[2,4,1,""],metadata:[2,4,1,""],price:[2,4,1,""],qty:[2,4,1,""],timestamp:[2,4,1,""]},"pyqstrat.pyqstrat_cpp.QuoteTOBAggregator":{__call__:[2,2,1,""],__init__:[2,2,1,""],close:[2,2,1,""]},"pyqstrat.pyqstrat_cpp.Record":{__init__:[2,4,1,""]},"pyqstrat.pyqstrat_cpp.RecordFieldParser":{__call__:[2,2,1,""],__init__:[2,2,1,""]},"pyqstrat.pyqstrat_cpp.RecordFilter":{__call__:[2,2,1,""],__init__:[2,2,1,""]},"pyqstrat.pyqstrat_cpp.RecordGenerator":{__call__:[2,2,1,""],__init__:[2,2,1,""]},"pyqstrat.pyqstrat_cpp.RecordParser":{__init__:[2,2,1,""],add_line:[2,2,1,""],parse:[2,2,1,""]},"pyqstrat.pyqstrat_cpp.RegExLineFilter":{__call__:[2,2,1,""],__init__:[2,2,1,""]},"pyqstrat.pyqstrat_cpp.Schema":{BOOL:[2,4,1,""],FLOAT32:[2,4,1,""],FLOAT64:[2,4,1,""],INT32:[2,4,1,""],INT64:[2,4,1,""],STRING:[2,4,1,""],TIMESTAMP_MICRO:[2,4,1,""],TIMESTAMP_MILLI:[2,4,1,""],Type:[2,1,1,""],__init__:[2,2,1,""],types:[2,4,1,""]},"pyqstrat.pyqstrat_cpp.Schema.Type":{BOOL:[2,4,1,""],FLOAT32:[2,4,1,""],FLOAT64:[2,4,1,""],INT32:[2,4,1,""],INT64:[2,4,1,""],STRING:[2,4,1,""],TIMESTAMP_MICRO:[2,4,1,""],TIMESTAMP_MILLI:[2,4,1,""],__init__:[2,2,1,""]},"pyqstrat.pyqstrat_cpp.SubStringLineFilter":{__call__:[2,2,1,""],__init__:[2,2,1,""]},"pyqstrat.pyqstrat_cpp.TextFileDecompressor":{__call__:[2,2,1,""],__init__:[2,2,1,""]},"pyqstrat.pyqstrat_cpp.TextFileProcessor":{__call__:[2,2,1,""],__init__:[2,2,1,""]},"pyqstrat.pyqstrat_cpp.TextOpenInterestParser":{__call__:[2,2,1,""],__init__:[2,2,1,""]},"pyqstrat.pyqstrat_cpp.TextOtherParser":{__call__:[2,2,1,""],__init__:[2,2,1,""]},"pyqstrat.pyqstrat_cpp.TextQuotePairParser":{__call__:[2,2,1,""],__init__:[2,2,1,""]},"pyqstrat.pyqstrat_cpp.TextQuoteParser":{__call__:[2,2,1,""],__init__:[2,2,1,""]},"pyqstrat.pyqstrat_cpp.TextRecordParser":{__init__:[2,2,1,""]},"pyqstrat.pyqstrat_cpp.TextTradeParser":{__call__:[2,2,1,""],__init__:[2,2,1,""]},"pyqstrat.pyqstrat_cpp.TimestampParser":{__call__:[2,2,1,""],__init__:[2,2,1,""]},"pyqstrat.pyqstrat_cpp.TradeBarAggregator":{__call__:[2,2,1,""],__init__:[2,2,1,""],close:[2,2,1,""]},"pyqstrat.pyqstrat_cpp.TradeRecord":{__init__:[2,2,1,""],id:[2,4,1,""],metadata:[2,4,1,""],price:[2,4,1,""],qty:[2,4,1,""],timestamp:[2,4,1,""]},"pyqstrat.pyqstrat_cpp.Writer":{__init__:[2,4,1,""],close:[2,2,1,""],write_batch:[2,2,1,""]},"pyqstrat.pyqstrat_cpp.WriterCreator":{__call__:[2,2,1,""],__init__:[2,2,1,""]},"pyqstrat.pyqstrat_cpp.ostream_redirect":{__init__:[2,2,1,""]},"pyqstrat.strategy":{Strategy:[2,1,1,""],test_strategy:[2,3,1,""]},"pyqstrat.strategy.Strategy":{__init__:[2,2,1,""],add_indicator:[2,2,1,""],add_market_sim:[2,2,1,""],add_rule:[2,2,1,""],add_signal:[2,2,1,""],df_data:[2,2,1,""],df_orders:[2,2,1,""],df_pnl:[2,2,1,""],df_returns:[2,2,1,""],df_trades:[2,2,1,""],evaluate_returns:[2,2,1,""],orders:[2,2,1,""],plot:[2,2,1,""],plot_returns:[2,2,1,""],run:[2,2,1,""],run_indicators:[2,2,1,""],run_rules:[2,2,1,""],run_signals:[2,2,1,""],trades:[2,2,1,""]},"pyqstrat.trade_bars":{TradeBars:[2,1,1,""],roll_futures:[2,3,1,""],sort_trade_bars:[2,3,1,""],sort_trade_bars_key:[2,3,1,""],test_trade_bars:[2,3,1,""]},"pyqstrat.trade_bars.TradeBars":{__init__:[2,2,1,""],add_timestamps:[2,2,1,""],c:[2,4,1,""],describe:[2,2,1,""],df:[2,2,1,""],errors:[2,2,1,""],freq_str:[2,2,1,""],h:[2,4,1,""],has_ohlc:[2,2,1,""],l:[2,4,1,""],o:[2,4,1,""],overview:[2,2,1,""],plot:[2,2,1,""],resample:[2,2,1,""],time_distribution:[2,2,1,""],timestamp:[2,4,1,""],v:[2,4,1,""],valid_row:[2,2,1,""],vwap:[2,4,1,""],warnings:[2,2,1,""]},pyqstrat:{account:[2,0,0,"-"],evaluator:[2,0,0,"-"],holiday_calendars:[2,0,0,"-"],marketdata_processor:[2,0,0,"-"],optimize:[2,0,0,"-"],orders:[2,0,0,"-"],plot:[2,0,0,"-"],portfolio:[2,0,0,"-"],pq_types:[2,0,0,"-"],pq_utils:[2,0,0,"-"],pyqstrat_cpp:[2,0,0,"-"],strategy:[2,0,0,"-"],trade_bars:[2,0,0,"-"]}},objnames:{"0":["py","module","Python module"],"1":["py","class","Python class"],"2":["py","method","Python method"],"3":["py","function","Python function"],"4":["py","attribute","Python attribute"],"5":["py","staticmethod","Python static method"]},objtypes:{"0":"py:module","1":"py:class","2":"py:method","3":"py:function","4":"py:attribute","5":"py:staticmethod"},terms:{"05t00":2,"07t00":2,"09t00":2,"100k":2,"10t00":2,"15t15":2,"19t15":2,"2_quot":2,"abstract":2,"boolean":2,"case":2,"class":2,"default":2,"enum":2,"final":2,"float":2,"function":2,"import":2,"int":2,"long":2,"new":2,"return":2,"short":2,"static":2,"true":2,"try":2,For:2,One:2,The:2,There:2,These:2,Use:2,Used:2,Will:2,__call__:2,__init__:2,_next:2,_tmp:2,abbrevi:2,abc:2,about:2,abov:2,access:2,accord:2,account:1,account_timestamp:2,accur:2,across:2,act:2,action:2,actual:2,add:2,add_calendar:2,add_contract:2,add_ind:2,add_lin:2,add_market_sim:2,add_metr:2,add_pnl:2,add_record:2,add_rul:2,add_sign:2,add_strategi:2,add_timestamp:2,add_trad:2,add_trading_dai:2,adding:2,addit:2,adopt:2,after:2,against:2,agg:2,aggreg:2,aggregator_cr:2,algorithm:2,all:2,all_timestamp:2,allopeninterestaggreg:2,allotheraggreg:2,allow:2,allquoteaggreg:2,allquotepairaggreg:2,alltradeaggreg:2,along:2,also:2,amean:2,ani:2,annual:2,anoth:2,anyth:2,apach:2,appli:2,applic:2,appropri:2,approx:2,arang:2,arg:2,argument:2,arithmet:2,around:2,arrai:2,arri:2,arrow:2,arrow_writer_cr:2,arrowwrit:2,arrowwritercr:2,as_str:2,ascend:2,ask:2,ask_price_idx:2,ask_qty_idx:2,assert:2,assign:2,assum:2,attribut:2,averag:2,avoid:2,axes:2,axi:2,back:2,backtest:2,backtest_end:2,backward:2,bad_line_handl:2,badlinehandl:2,bar:2,base:2,base_d:2,base_date_filename_mapp:2,base_date_mapp:2,basic:2,batch:2,batch_by_id:2,batch_id:2,batch_siz:2,becaus:2,becom:2,been:2,befor:2,begin:2,being:2,below:2,besid:2,best:2,between:2,bid:2,bid_offer_idx:2,bid_price_idx:2,bid_qty_idx:2,bid_str:2,biggest:2,billion:2,bin:2,black:2,black_scholes_pric:2,blue:2,book:2,bool:2,both:2,bottom:2,boundari:2,box:2,boxplot:2,broken:2,broker:2,bubbl:2,bucket:2,bucket_nam:2,bucket_valu:2,bucketedvalu:2,bui:2,busday_offset:2,busi:2,bz2:2,calc:2,calc_net_pnl:2,calc_trade_pnl:2,calcul:2,calendar:2,calendar_nam:2,call:2,calmar:2,came:2,can:2,candlestick:2,cannot:2,captur:2,categor:2,categori:2,caus:2,cdf:2,cent:2,chang:2,charact:2,check:2,check_data_s:2,checkfield:2,clean:2,clear:2,clear_al:2,close:2,close_qti:2,closest:2,clutter:2,cmap:2,code:2,color:2,colordown:2,colormap:2,colorup:2,column:2,com:2,combin:2,come:2,command:2,commis:2,commiss:2,common:2,compon:2,compound:2,compress:2,comput:2,compute_amean:2,compute_annual_return:2,compute_bucketed_return:2,compute_calmar:2,compute_dates_3yr:2,compute_equ:2,compute_gmean:2,compute_mar:2,compute_maxdd_d:2,compute_maxdd_date_3yr:2,compute_maxdd_pct:2,compute_maxdd_pct_3yr:2,compute_maxdd_start:2,compute_maxdd_start_3yr:2,compute_metr:2,compute_num_period:2,compute_periods_per_year:2,compute_return_metr:2,compute_returns_3yr:2,compute_rolling_dd:2,compute_rolling_dd_3yr:2,compute_sharp:2,compute_sortino:2,compute_std:2,concanten:2,concaten:2,concurr:2,condit:2,condition_func:2,confid:2,consid:2,constant:2,construct:2,constructor:2,contain:2,content:[0,1],context:2,continu:2,contour:2,contract:2,contract_group:2,contractgroup:2,contractpnl:2,conveni:2,convent:2,convert:2,core:2,correl:2,correspond:2,cost:2,cost_func:2,could:2,count:2,cpu:2,creat:2,create_batch_id:2,create_batch_id_fil:2,create_text_file_processor:2,cross:2,crowd:2,csv:2,ctime:2,cubic:2,cumul:2,currenc:2,current:2,curv:2,custom:2,customari:2,dai:2,daili:2,darkgreen:2,dash:2,data:2,data_list:2,datafram:2,datatyp:2,date2num:2,date:2,date_2_num:2,date_format:2,date_func:2,date_lin:2,date_rang:2,dateformatt:2,datelin:2,dates2:2,datetim:2,datetime64:2,datfram:2,day_of_week_num:2,deal:2,debug:2,dec2018:2,decid:2,decim:2,decode_future_cod:2,defin:2,delimit:2,delta:2,densiti:2,depend:2,depends_on:2,depends_on_ind:2,depends_on_sign:2,describ:2,descript:2,detail:2,determin:2,deviat:2,df_account_pnl:2,df_data:2,df_display_max_column:2,df_display_max_row:2,df_experi:2,df_float_sf:2,df_order:2,df_pnl:2,df_return:2,df_trade:2,diagnos:2,dict:2,dictionari:2,differ:2,dir_fd:2,directli:2,directori:[0,2],dirnam:2,discard:2,discount:2,disk:2,displai:2,display_legend:2,display_return_metr:2,display_summari:2,distribut:2,divid:2,dividend:2,doctest:2,document:2,doe:2,doesn:2,dollar:2,don:2,done:2,dont:2,down:2,downsampl:2,draw:2,draw_3d_plot:2,draw_boxplot:2,draw_candlestick:2,draw_date_lin:2,draw_horizontal_lin:2,draw_poli:2,draw_vertical_lin:2,drawdown:2,dtype:2,each:2,earlier:2,easier:2,easili:2,edg:2,edgecolor:2,either:2,element:2,ellipsi:2,els:2,empti:2,end:2,end_dat:2,enter:2,enter_long:2,enter_short:2,entri:2,eof:2,epoch:2,equal:2,equiti:2,error:2,esh9:2,etc:2,eurex:2,euroepean:2,european:2,evalu:1,evaluate_return:2,even:2,everi:2,everyth:2,exampl:2,except:2,exchang:2,exchange_nam:2,exclud:2,exclude_pattern:2,exclus:2,execut:2,exist:2,exit:2,exit_long:2,exit_short:2,exp:2,experi:2,experiment_list:2,expir:2,expiri:2,explan:2,express:2,extens:2,extra:2,f2583e:2,facecolor:2,fail:2,fall:2,fals:2,faster:2,fee:2,few:2,field:2,field_nam:2,fig:2,figsiz:2,figur:2,file:2,file_nam:2,file_processor:2,file_processor_cr:2,filenam:2,filepath:2,fileprocessor:2,fill:2,fill_valu:2,filter:2,filterwarn:2,find:2,find_index_befor:2,find_last_non_nan_index:2,finit:2,first:2,fix:2,fixedwidthtimepars:2,fixedwithtimepars:2,flag:2,flag_idx:2,flag_valu:2,float32:2,float64:2,float_precis:2,flush:2,fmt:2,fname:2,follow:2,format:2,formatt:2,formattimestamppars:2,formula:2,forward:2,fraction:2,free:2,freq:2,freq_str:2,frequenc:2,fridai:2,from:2,full:2,func:2,fut_pric:2,futur:2,future_cod:2,gamma:2,gap:2,gener:2,geometr:2,get:[0,2],get_all_timestamp:2,get_calendar:2,get_contract:2,get_date_formatt:2,get_empty_np_valu:2,get_field_indic:2,get_fut_cod:2,get_temp_dir:2,get_trading_dai:2,gettempdir:2,ggplot:2,give:2,given:2,global:2,gmean:2,goe:2,good:2,graph:2,greater:2,green:2,grid:2,griddata:2,group:2,guarante:2,gzip:2,had:2,half:2,handle_non_finite_return:2,has:2,has_displai:2,has_ohlc:2,have:2,header:2,header_parser_cr:2,header_record_gener:2,heartbeat:2,hedg:2,height:2,height_ratio:2,hello:2,help:2,helper:2,here:2,high:2,higher:2,highest_cost:2,hole:2,holidai:2,holiday_calendar:1,home:2,horizont:2,horizontal_lin:2,horizontallin:2,hour:2,hours_siz:2,hours_start:2,how:2,hspace:2,http:2,hundredth:2,ibm:2,id_field_indic:2,identifi:2,ignor:2,iloc:2,implement:2,impli:2,implied_vol:2,incept:2,incl:2,includ:2,include_first:2,include_last:2,include_pattern:2,incorrect:2,increas:2,increment:2,index:[0,2],indic:2,indicator_nam:2,indicator_properti:2,inf:2,infer:2,infer_compress:2,infer_frequ:2,infin:2,info:2,inform:2,init:2,initi:2,initial_metr:2,inplac:2,input:2,input_file_path:2,input_filenam:2,input_filename_provid:2,input_filepath:2,instead:2,instrument:2,int32:2,int64:2,integ:2,interact:2,interest:2,interfac:2,intern:2,interpol:2,interv:2,invalid:2,is_new:2,is_open_interest:2,is_oth:2,is_quot:2,is_quote_pair:2,is_trad:2,is_trading_dai:2,isclos:2,isfieldinlist:2,isnan:2,item:2,iter:2,its:2,itself:2,jun2018:2,just:2,keep:2,kei:2,kind:2,kwarg:2,label:2,lambda:2,larg:2,last:2,last_update_tim:2,later:2,lead:2,leading_nan_to_zero:2,leading_non_finite_to_zero:2,least:2,left:2,leg:2,legend:2,legend_loc:2,len:2,length:2,less:2,letter:2,level:2,like:2,limit:2,limit_pric:2,limitord:2,line:2,line_filt:2,line_numb:2,line_typ:2,line_width:2,linear:2,linear_interpol:2,linefilt:2,linestyl:2,list:2,locat:2,lock:2,log:2,log_i:2,logarithm:2,logic:2,look:2,lookup:2,loss:2,low:2,lower:2,lowercas:2,lowest_cost:2,lzip:2,machin:2,mai:2,main:[0,2],make:2,make_lowercas:2,mani:2,map:2,mar2018:2,mar:2,march:2,marker:2,marker_color:2,marker_prop:2,marker_properti:2,marker_s:2,market:2,market_sim_funct:2,marketdata_processor:1,marketord:2,match:2,math:2,matplotlib:2,matur:2,max:2,max_batch_s:2,max_process:2,mdate:2,mdd_date:2,mdd_date_3yr:2,mdd_pct:2,mdd_pct_3yr:2,mean:2,measur:2,median:2,member:2,memori:2,met:2,meta:2,meta_field_indic:2,metadata:2,metric:2,metric_nam:2,micro:2,micros_s:2,micros_start:2,microsec:2,microsecond:2,midnight:2,might:2,milli:2,millis_s:2,millis_since_epoch:2,millis_start:2,millisec:2,millisecond:2,min:2,mini:2,minut:2,minutes_s:2,minutes_start:2,miss:2,missing_data_handl:2,missingdatahandl:2,mode:2,modfic:2,modifi:2,modifiedfollow:2,modifiedpreced:2,modul:[0,1],mondai:2,monoton:2,monotonically_increas:2,month:2,more:2,most:2,move:2,mpl_figsiz:2,multipl:2,multipli:2,multiprocess:2,must:2,name:2,nan:2,nan_to_zero:2,nat:2,ndarrai:2,nearest:2,necessari:2,need:2,neg:2,net:2,net_pnl:2,new_pric:2,new_qti:2,new_timestamp:2,newer:2,next:2,non:2,none:2,nonzero:2,normal:2,notat:2,notch:2,note:2,nov:2,np_dtype:2,np_find_closest:2,np_get_index:2,np_seterr:2,num_dai:2,num_process:2,num_trading_dai:2,number:2,numpi:2,nyse:2,obj:2,object:2,obtain:2,off:2,offer:2,offer_str:2,ohlc:2,ohlcv:2,one:2,ones:2,onli:2,open:2,open_pric:2,open_qti:2,openinterestrecord:2,oppos:2,optim:1,option:2,order:1,orderstatu:2,origin:2,ostream_redirect:2,other:2,other_cost:2,otherrecord:2,otherwis:2,out:2,outlier:2,output:2,output_dir:2,output_file_1:2,output_file_prefix:2,output_file_prefix_mapp:2,outsid:2,over:2,overview:2,own:2,packag:[0,1],page:0,paid:2,pair:2,panda:2,param:2,paramet:2,parrallel:2,pars:2,parser:2,part:2,pass:2,past:2,path:2,pathfilenameprovid:2,pattern:2,per:2,percent:2,percentag:2,percentil:2,percentile_of_scor:2,percentileofscor:2,performancewarn:2,period:2,periods_per_year:2,perman:2,place:2,pleas:0,plot:1,plot_2d:2,plot_3d:2,plot_equ:2,plot_return:2,plot_return_metr:2,plot_styl:2,plot_timestamp:2,plot_typ:2,plt:2,plu:2,pnl:2,pnl_calc_tim:2,pnl_column:2,point:2,polygram:2,portfolio:1,posit:2,position_filt:2,possibl:2,potenti:2,pq_type:1,pq_util:1,pre:2,preced:2,predefin:2,prefer:2,premium:2,prepend:2,present:2,previous:2,price:2,price_funct:2,price_idx:2,price_multipli:2,price_qty_missing_data_handl:2,priceqtymissingdatahandl:2,primari:2,primary_ind:2,primary_indicators_dual_axi:2,print:2,print_time_distribut:2,printbadlinehandl:2,problem:2,process:2,process_marketdata:2,process_marketdata_fil:2,properti:2,proport:2,proportional_width:2,provid:2,put:2,pybind11_builtin:2,pybind11_object:2,pyqstrat_cpp:1,python:2,qty:2,qty_idx:2,quantil:2,quantiti:2,question:2,queu:2,quot:2,quote_pair:2,quotepairrecord:2,quoterecord:2,quotetobaggreg:2,rais:2,raise_on_error:2,rand:2,random:2,rang:2,rate:2,ratio:2,reach:2,read:[0,2],read_holidai:2,readi:2,readm:0,realiz:2,realli:2,reason:2,reason_cod:2,reasoncod:2,recomput:2,record:2,record_filt:2,record_gener:2,record_pars:2,record_parser_cr:2,recordfieldpars:2,recordfilt:2,recordgener:2,recordpars:2,red:2,ref_filenam:2,refer:2,regex:2,regexlinefilt:2,regular:2,reindex:2,relev:2,remain:2,rememb:2,remov:2,remove_missing_properti:2,renam:2,reopen_qti:2,replac:2,replic:2,report:2,reprent:2,repres:2,resampl:2,resample_func:2,resample_t:2,resample_trade_bar:2,resample_vwap:2,rest:2,restart:2,restrict:2,result:2,ret:2,retriev:2,return_full_df:2,returns_3yr:2,rho:2,right:2,risk:2,roll:2,roll_flag:2,roll_futur:2,rolling_dd:2,rolling_dd_3yr:2,rolling_dd_3yr_timestamp:2,rolling_dd_d:2,rollord:2,round:2,row:2,rst:0,rule:2,rule_funct:2,rule_nam:2,run:2,run_final_calc:2,run_ind:2,run_rul:2,run_sign:2,same:2,sampl:2,sampling_frequ:2,save:2,scalar:2,scatter:2,schema:2,schole:2,scientif:2,scipi:2,search:0,second:2,secondari:2,secondary_i:2,secondary_ind:2,secondary_indicators_dual_axi:2,seconds_s:2,seconds_start:2,section:2,see:2,select:2,self:2,sell:2,sep2018:2,separ:2,sequenc:2,seri:2,series_to_arrai:2,server:2,set:2,set_default:2,set_index:2,set_printopt:2,seterr:2,shape:2,share:2,sharp:2,sharpe0:2,shift:2,shift_np:2,should:2,show:2,show_al:2,show_date_gap:2,show_grid:2,show_mean:2,show_outli:2,shown:2,sig_true_valu:2,sigma:2,sign:2,signal:2,signal_funct:2,signal_nam:2,signal_properti:2,signatur:2,signific:2,similar:2,simpl:2,simplenamespac:2,simul:2,sinc:2,singl:2,singledirectoryfilenamemapp:2,size:2,skip:2,skip_row:2,sleep:2,slot:2,slow:2,smaller:2,solid:2,some:2,someth:2,sometim:2,sort:2,sort_column:2,sort_ord:2,sort_trade_bar:2,sort_trade_bars_kei:2,sorted_dict:2,sourc:2,space:2,special:2,specif:2,specifi:2,spot:2,spx_2018:2,spy_1970:2,stack:2,stackoverflow:2,standard:2,start:[0,2],start_dat:2,starting_equ:2,stat:2,statu:2,std:2,stderr:2,stdout:2,stop:2,stoplimitord:2,storag:2,store:2,str2date:2,str:2,strategi:1,strategy_context:2,strategy_nam:2,strftime:2,strike:2,string:2,strip:2,strip_id:2,strip_meta:2,strtup2dat:2,style:2,sub:2,subclass:2,submodul:1,subplot:2,subplot_list:2,subsampl:2,subsequ:2,subsequent_non_finite_to_zero:2,substringlinefilt:2,success:2,successfulli:2,suffix:2,suggest:2,suit:2,sum:2,summar:2,sundai:2,suppli:2,sure:2,surfac:2,symbol:2,tabl:2,take:2,taller:2,target:2,tell:2,temp:2,temp_dir:2,tempfil:2,temporari:2,test:2,test_account:2,test_evalu:2,test_optim:2,test_plot:2,test_strategi:2,test_trade_bar:2,text:2,text_file_record_gener:2,textfiledecompressor:2,textfileprocessor:2,textheaderpars:2,textopeninterestpars:2,textotherpars:2,textquotepairpars:2,textquotepars:2,textrecordpars:2,texttradepars:2,than:2,thei:2,them:2,therefor:2,theta:2,thi:2,think:2,those:2,threshold:2,through:2,tick:2,ticker:2,time:2,time_distribut:2,time_distribution_frequ:2,time_format:2,timedelta64:2,timeseri:2,timestamp:2,timestamp_indic:2,timestamp_micro:2,timestamp_milli:2,timestamp_pars:2,timestamp_unit:2,timestamppars:2,titl:2,tmp:2,to_csv:2,togeth:2,too:2,top:2,touch:2,trace:2,track:2,trade:2,trade_bar:1,trade_lag:2,trade_marker_properti:2,trade_sets_by_reason_cod:2,tradebar:2,tradebaraggreg:2,tradebarseri:2,traderecord:2,tradeset:2,treat:2,trigger:2,trigger_pric:2,tup:2,tupl:2,turn:2,two:2,txt:2,type:2,uess:2,unchang:2,uncompress:2,uncorrel:2,underli:2,uniqu:2,unit:2,unix:2,unless:2,unreal:2,unwritten:2,updat:2,use:2,used:2,useful:2,using:2,usual:2,util:2,v_next:2,valid:2,valid_row:2,valu:2,variabl:2,vector:2,vega:2,versa:2,vertic:2,vertical_lin:2,verticallin:2,vice:2,view:2,viridi:2,volatil:2,volum:2,vwap:2,wai:2,want:2,warmup:2,warn:2,warn_std:2,week:2,weekend:2,weight:2,well:2,were:2,when:2,whenev:2,where:2,whether:2,which:2,whisker:2,whitespac:2,whole:2,width:2,within:2,without:2,work:2,would:2,write:2,write_batch:2,writer:2,writer_cr:2,writercr:2,written:2,xlabel:2,xlim:2,xxx:2,xydata:2,xyz:2,xyzdata:2,xzy:2,y_tick_format:2,year:2,yield:2,ylabel:2,ylim:2,you:2,your:2,zero:2,zero_to_nan:2,zlabel:2,zorder:2},titles:["API documentation for pyqstrat","pyqstrat","pyqstrat package"],titleterms:{account:2,api:0,content:2,document:0,evalu:2,holiday_calendar:2,indic:0,marketdata_processor:2,modul:2,optim:2,order:2,packag:2,plot:2,portfolio:2,pq_type:2,pq_util:2,pyqstrat:[0,1,2],pyqstrat_cpp:2,strategi:2,submodul:2,tabl:0,trade_bar:2}})