STYLES = {
    'siderbar':{
        "position": "fixed",      # 固定定位（不随页面滚动）
        "top": 0,                 # 顶部对齐
        "left": 0,                # 左侧对齐
        # "bottom": 0,             # 底部对齐（实现全高度）
        "width": "17%",         # 宽度20rem（约320px）
        "padding": "0.3rem 0.3rem",   # 内边距上下2rem，左右1rem
        "background-color": "#f8f9fa",  # 浅灰色背景
        "overflow": "auto",       # 内容溢出时显示滚动条
        "box-shadow": "2px 0px 5px rgba(0,0,0,0.1)"  # 右侧阴影
    },
    'hideden_siderbar':{
        "position": "fixed",
        "top": 0,
        "left": 0,
        "bottom": 0,
        "width": "20rem",
        "padding": "2rem 1rem",
        "background-color": "#f8f9fa",
        "overflow": "auto",
        "box-shadow": "2px 0px 5px rgba(0,0,0,0.1)",
        "margin-left": "-22rem",  # 向左移出视口（隐藏）
        "transition": "margin-left 0.5s ease"  # 过渡动画效果
    },
    'contents' : {
        # 'display': 'grid',
        # 'gridTemplateColumns': '1fr 1fr',
        # 'gap': '2rem',
        # 'marginBottom': '2rem',
        # "height": "100%",
        # "overflow": "hidden"
        "margin-left": "17.5%",
        "margin-right": "0.5rem",
        # 'marginTop': '2rem',
        "padding": "0rem",
        "transition": "margin-left 0.5s ease",
        "background-color": "#fdfdfd",
        # 'gridTemplateColumns': 'repeat(auto-fit, minmax(100, 1fr))',  # 自适应列
    },
    'hideden_contents' : {
        "margin-left": "0.5rem",
        "margin-right": "0.5rem",
        "padding": "0rem",
        "transition": "margin-left 0.5s ease",
        "background-color": "#fdfdfd",
    },
    'collapse_button_style':{
        "position": "fixed",       # 固定定位
        "top": "10px",             # 距离顶部10px
        "left": "0",               # 贴左对齐
        "zIndex": 1e6,            # 确保在最上层
        "width": "30px",           # 按钮宽度
        "height": "30px",         # 按钮高度
        "border-radius": "50%",   # 圆形按钮
        "border": "1px solid #ddd",# 浅灰色边框
        "background-color": "#E14B0A",  # 背景色与侧边栏一致
        "box-shadow": "2px 2px 6px rgba(0, 0, 0, 0.1)",  # 立体阴影
        "transition": "margin-left 0.5s ease",
        "display": "flex",
        "justify-content": "center",  # 水平居中
        "align-items": "center",   # 垂直居中
        "cursor": "pointer",       # 鼠标手型指针
    },
    'expand_button_style':{
        "position": "fixed",       # 固定定位
        "top": "10px",             # 距离顶部10px
        "left": "0",               # 贴左对齐
        "zIndex": 1e6,            # 确保在最上层
        "width": "30px",           # 按钮宽度
        "height": "30px",         # 按钮高度
        "border-radius": "50%",   # 圆形按钮
        "border": "1px solid #ddd",# 浅灰色边框
        "background-color": "#3BDEC5",  # 背景色与侧边栏一致
        "box-shadow": "2px 2px 6px rgba(0, 0, 0, 0.1)",  # 立体阴影
        "transition": "margin-left 0.5s ease",
        "display": "none",         # 默认隐藏
        "justify-content": "center",  # 水平居中
        "align-items": "center",   # 垂直居中
        "cursor": "pointer",       # 鼠标手型指针
    },

    'layout': {
        'padding': '2rem',        # 内边距
        'margin': '0 auto',        # 水平居中
        'backgroundColor': '#f5f6fa'  # 浅灰色背景
    },
    'controlGroup': {
        'display': 'grid',         # 网格布局
        'gridTemplateColumns': 'repeat(auto-fit, minmax(240px, 1fr))',  # 自适应列
        'gap': '1rem',             # 元素间距
        'marginBottom': '1.5rem'   # 底部外边距
    },
    'graphContainer': {
        "display": "flex",         # 弹性布局
        "height": "calc(100vh - 4rem)",  # 视口高度减4rem
        "background-color": "#fff",# 白色背景
        "border-radius": "8px",    # 圆角
        "box-shadow": "0 4px 8px rgba(0, 0, 0, 0.17)",  # 阴影
        "overflow": "hidden"       # 内容裁剪
    },
    'floatingControls': {  # 悬浮操作按钮组
        'position': 'fixed',    # 绝对定位
        'top': '1rem',             # 距顶部1rem
        'right': '1rem',           # 距右侧1rem
        'zIndex': 100,             # 层级高于图表
        'display': 'flex',         # 弹性布局
        'gap': '0.5rem'            # 按钮间距
    },
    'iconButton': {  
        'width': '2.5rem',        
        'height': '2.5rem',      
        'borderRadius': '50%',   
        'border': 'none',        
        'cursor': 'pointer',    
        "overflow": "hidden", 
        "justify-content": "center",  
        "align-items": "center", 
        "display": "flex",  
        'flexShrink': 0, 
        'flexGrow': 0, 
        'transition': 'all 0.5s'
    }
}