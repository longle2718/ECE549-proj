




<!DOCTYPE html>
<html class="   ">
  <head prefix="og: http://ogp.me/ns# fb: http://ogp.me/ns/fb# object: http://ogp.me/ns/object# article: http://ogp.me/ns/article# profile: http://ogp.me/ns/profile#">
    <meta charset='utf-8'>
    <meta http-equiv="X-UA-Compatible" content="IE=edge">
    
    
    <title>ECE549-proj/clustering.m at master · long0612/ECE549-proj</title>
    <link rel="search" type="application/opensearchdescription+xml" href="/opensearch.xml" title="GitHub" />
    <link rel="fluid-icon" href="https://github.com/fluidicon.png" title="GitHub" />
    <link rel="apple-touch-icon" sizes="57x57" href="/apple-touch-icon-114.png" />
    <link rel="apple-touch-icon" sizes="114x114" href="/apple-touch-icon-114.png" />
    <link rel="apple-touch-icon" sizes="72x72" href="/apple-touch-icon-144.png" />
    <link rel="apple-touch-icon" sizes="144x144" href="/apple-touch-icon-144.png" />
    <meta property="fb:app_id" content="1401488693436528"/>

      <meta content="@github" name="twitter:site" /><meta content="summary" name="twitter:card" /><meta content="long0612/ECE549-proj" name="twitter:title" /><meta content="ECE549-proj - ECE 549 Final Project" name="twitter:description" /><meta content="https://avatars2.githubusercontent.com/u/5008450?s=400" name="twitter:image:src" />
<meta content="GitHub" property="og:site_name" /><meta content="object" property="og:type" /><meta content="https://avatars2.githubusercontent.com/u/5008450?s=400" property="og:image" /><meta content="long0612/ECE549-proj" property="og:title" /><meta content="https://github.com/long0612/ECE549-proj" property="og:url" /><meta content="ECE549-proj - ECE 549 Final Project" property="og:description" />

    <link rel="assets" href="https://github.global.ssl.fastly.net/">
    <link rel="conduit-xhr" href="https://ghconduit.com:25035/">
    <link rel="xhr-socket" href="/_sockets" />

    <meta name="msapplication-TileImage" content="/windows-tile.png" />
    <meta name="msapplication-TileColor" content="#ffffff" />
    <meta name="selected-link" value="repo_source" data-pjax-transient />
    <meta content="collector.githubapp.com" name="octolytics-host" /><meta content="collector-cdn.github.com" name="octolytics-script-host" /><meta content="github" name="octolytics-app-id" /><meta content="827EFF8F:545B:75D925:537125ED" name="octolytics-dimension-request_id" /><meta content="5008450" name="octolytics-actor-id" /><meta content="long0612" name="octolytics-actor-login" /><meta content="5c749740fbf2dca7f89f8380991e36a5e7abb1d36a4d46c5481f996f0f3c8a88" name="octolytics-actor-hash" />
    

    
    
    <link rel="icon" type="image/x-icon" href="https://github.global.ssl.fastly.net/favicon.ico" />

    <meta content="authenticity_token" name="csrf-param" />
<meta content="hZtBJd7S+/picI/LXHoN7VcNiON7O+t4iF0BTl5DPmnzQFkiw1Saub7npAxsPLlbFH7fG974nn4rnDIXFd9B4g==" name="csrf-token" />

    <link href="https://github.global.ssl.fastly.net/assets/github-58e181c5cf58206dac2f13d435da7a71ca165593.css" media="all" rel="stylesheet" type="text/css" />
    <link href="https://github.global.ssl.fastly.net/assets/github2-1a3c410b868af7031a33d9c381adc57fbdd76b68.css" media="all" rel="stylesheet" type="text/css" />
    


    <meta http-equiv="x-pjax-version" content="8d3812e005f9ff2b254914e5f873c6f0">

      
  <meta name="description" content="ECE549-proj - ECE 549 Final Project" />

  <meta content="5008450" name="octolytics-dimension-user_id" /><meta content="long0612" name="octolytics-dimension-user_login" /><meta content="18545058" name="octolytics-dimension-repository_id" /><meta content="long0612/ECE549-proj" name="octolytics-dimension-repository_nwo" /><meta content="true" name="octolytics-dimension-repository_public" /><meta content="false" name="octolytics-dimension-repository_is_fork" /><meta content="18545058" name="octolytics-dimension-repository_network_root_id" /><meta content="long0612/ECE549-proj" name="octolytics-dimension-repository_network_root_nwo" />
  <link href="https://github.com/long0612/ECE549-proj/commits/master.atom" rel="alternate" title="Recent Commits to ECE549-proj:master" type="application/atom+xml" />

  </head>


  <body class="logged_in  env-production windows vis-public page-blob">
    <a href="#start-of-content" tabindex="1" class="accessibility-aid js-skip-to-content">Skip to content</a>
    <div class="wrapper">
      
      
      
      


      <div class="header header-logged-in true">
  <div class="container clearfix">

    <a class="header-logo-invertocat" href="https://github.com/">
  <span class="mega-octicon octicon-mark-github"></span>
</a>

    
    <a href="/notifications" aria-label="You have no unread notifications" class="notification-indicator tooltipped tooltipped-s" data-hotkey="g n">
        <span class="mail-status all-read"></span>
</a>

      <div class="command-bar js-command-bar  in-repository">
          <form accept-charset="UTF-8" action="/search" class="command-bar-form" id="top_search_form" method="get">

<div class="commandbar">
  <span class="message"></span>
  <input type="text" data-hotkey="s, /" name="q" id="js-command-bar-field" placeholder="Search or type a command" tabindex="1" autocapitalize="off"
    
    data-username="long0612"
      data-repo="long0612/ECE549-proj"
      data-branch="master"
      data-sha="fc294bf3bdedf765709176f0e75e618c821bee13"
  >
  <div class="display hidden"></div>
</div>

    <input type="hidden" name="nwo" value="long0612/ECE549-proj" />

    <div class="select-menu js-menu-container js-select-menu search-context-select-menu">
      <span class="minibutton select-menu-button js-menu-target" role="button" aria-haspopup="true">
        <span class="js-select-button">This repository</span>
      </span>

      <div class="select-menu-modal-holder js-menu-content js-navigation-container" aria-hidden="true">
        <div class="select-menu-modal">

          <div class="select-menu-item js-navigation-item js-this-repository-navigation-item selected">
            <span class="select-menu-item-icon octicon octicon-check"></span>
            <input type="radio" class="js-search-this-repository" name="search_target" value="repository" checked="checked" />
            <div class="select-menu-item-text js-select-button-text">This repository</div>
          </div> <!-- /.select-menu-item -->

          <div class="select-menu-item js-navigation-item js-all-repositories-navigation-item">
            <span class="select-menu-item-icon octicon octicon-check"></span>
            <input type="radio" name="search_target" value="global" />
            <div class="select-menu-item-text js-select-button-text">All repositories</div>
          </div> <!-- /.select-menu-item -->

        </div>
      </div>
    </div>

  <span class="help tooltipped tooltipped-s" aria-label="Show command bar help">
    <span class="octicon octicon-question"></span>
  </span>


  <input type="hidden" name="ref" value="cmdform">

</form>
        <ul class="top-nav">
          <li class="explore"><a href="/explore">Explore</a></li>
            <li><a href="https://gist.github.com">Gist</a></li>
            <li><a href="/blog">Blog</a></li>
          <li><a href="https://help.github.com">Help</a></li>
        </ul>
      </div>

    


  <ul id="user-links">
    <li>
      <a href="/long0612" class="name">
        <img alt="Long Le" class=" js-avatar" data-user="5008450" height="20" src="https://avatars1.githubusercontent.com/u/5008450?s=140" width="20" /> long0612
      </a>
    </li>

    <li class="new-menu dropdown-toggle js-menu-container">
      <a href="#" class="js-menu-target tooltipped tooltipped-s" aria-label="Create new...">
        <span class="octicon octicon-plus"></span>
        <span class="dropdown-arrow"></span>
      </a>

      <div class="new-menu-content js-menu-content">
      </div>
    </li>

    <li>
      <a href="/settings/profile" id="account_settings"
        class="tooltipped tooltipped-s"
        aria-label="Account settings ">
        <span class="octicon octicon-tools"></span>
      </a>
    </li>
    <li>
      <form class="logout-form" action="/logout" method="post">
        <button class="sign-out-button tooltipped tooltipped-s" aria-label="Sign out">
          <span class="octicon octicon-log-out"></span>
        </button>
      </form>
    </li>

  </ul>

<div class="js-new-dropdown-contents hidden">
  

<ul class="dropdown-menu">
  <li>
    <a href="/new"><span class="octicon octicon-repo-create"></span> New repository</a>
  </li>
  <li>
    <a href="/organizations/new"><span class="octicon octicon-organization"></span> New organization</a>
  </li>


    <li class="section-title">
      <span title="long0612/ECE549-proj">This repository</span>
    </li>
      <li>
        <a href="/long0612/ECE549-proj/issues/new"><span class="octicon octicon-issue-opened"></span> New issue</a>
      </li>
      <li>
        <a href="/long0612/ECE549-proj/settings/collaboration"><span class="octicon octicon-person-add"></span> New collaborator</a>
      </li>
</ul>

</div>


    
  </div>
</div>

      

        



      <div id="start-of-content" class="accessibility-aid"></div>
          <div class="site" itemscope itemtype="http://schema.org/WebPage">
    <div id="js-flash-container">
      
    </div>
    <div class="pagehead repohead instapaper_ignore readability-menu">
      <div class="container">
        

<ul class="pagehead-actions">

    <li class="subscription">
      <form accept-charset="UTF-8" action="/notifications/subscribe" class="js-social-container" data-autosubmit="true" data-remote="true" method="post"><div style="margin:0;padding:0;display:inline"><input name="authenticity_token" type="hidden" value="QonVDx+NlAkEaWq6B98uDYVFcOD4b0AJ4fh2dpaP7ZhsdJE/QEa3HWiwKuwt4NAGAJf9VGn1Q40QTCHY2wpdiA==" /></div>  <input id="repository_id" name="repository_id" type="hidden" value="18545058" />

    <div class="select-menu js-menu-container js-select-menu">
      <a class="social-count js-social-count" href="/long0612/ECE549-proj/watchers">
        2
      </a>
      <span class="minibutton select-menu-button with-count js-menu-target" role="button" tabindex="0" aria-haspopup="true">
        <span class="js-select-button">
          <span class="octicon octicon-eye-unwatch"></span>
          Unwatch
        </span>
      </span>

      <div class="select-menu-modal-holder">
        <div class="select-menu-modal subscription-menu-modal js-menu-content" aria-hidden="true">
          <div class="select-menu-header">
            <span class="select-menu-title">Notification status</span>
            <span class="octicon octicon-remove-close js-menu-close"></span>
          </div> <!-- /.select-menu-header -->

          <div class="select-menu-list js-navigation-container" role="menu">

            <div class="select-menu-item js-navigation-item " role="menuitem" tabindex="0">
              <span class="select-menu-item-icon octicon octicon-check"></span>
              <div class="select-menu-item-text">
                <input id="do_included" name="do" type="radio" value="included" />
                <h4>Not watching</h4>
                <span class="description">You only receive notifications for conversations in which you participate or are @mentioned.</span>
                <span class="js-select-button-text hidden-select-button-text">
                  <span class="octicon octicon-eye-watch"></span>
                  Watch
                </span>
              </div>
            </div> <!-- /.select-menu-item -->

            <div class="select-menu-item js-navigation-item selected" role="menuitem" tabindex="0">
              <span class="select-menu-item-icon octicon octicon octicon-check"></span>
              <div class="select-menu-item-text">
                <input checked="checked" id="do_subscribed" name="do" type="radio" value="subscribed" />
                <h4>Watching</h4>
                <span class="description">You receive notifications for all conversations in this repository.</span>
                <span class="js-select-button-text hidden-select-button-text">
                  <span class="octicon octicon-eye-unwatch"></span>
                  Unwatch
                </span>
              </div>
            </div> <!-- /.select-menu-item -->

            <div class="select-menu-item js-navigation-item " role="menuitem" tabindex="0">
              <span class="select-menu-item-icon octicon octicon-check"></span>
              <div class="select-menu-item-text">
                <input id="do_ignore" name="do" type="radio" value="ignore" />
                <h4>Ignoring</h4>
                <span class="description">You do not receive any notifications for conversations in this repository.</span>
                <span class="js-select-button-text hidden-select-button-text">
                  <span class="octicon octicon-mute"></span>
                  Stop ignoring
                </span>
              </div>
            </div> <!-- /.select-menu-item -->

          </div> <!-- /.select-menu-list -->

        </div> <!-- /.select-menu-modal -->
      </div> <!-- /.select-menu-modal-holder -->
    </div> <!-- /.select-menu -->

</form>
    </li>

  <li>
  

  <div class="js-toggler-container js-social-container starring-container ">

    <form accept-charset="UTF-8" action="/long0612/ECE549-proj/unstar" class="js-toggler-form starred" data-remote="true" method="post"><div style="margin:0;padding:0;display:inline"><input name="authenticity_token" type="hidden" value="aQnhg+A02PWERctBNCJFgkhpgqV9bUwStWbipL1vo/O/v18cFZywonqzAj7xiizwl5fPV92A+e0VS3l9upSKVQ==" /></div>
      <button
        class="minibutton with-count js-toggler-target star-button"
        aria-label="Unstar this repository" title="Unstar long0612/ECE549-proj">
        <span class="octicon octicon-star-delete"></span><span class="text">Unstar</span>
      </button>
        <a class="social-count js-social-count" href="/long0612/ECE549-proj/stargazers">
          0
        </a>
</form>
    <form accept-charset="UTF-8" action="/long0612/ECE549-proj/star" class="js-toggler-form unstarred" data-remote="true" method="post"><div style="margin:0;padding:0;display:inline"><input name="authenticity_token" type="hidden" value="c1G+d3YU90Qevn0l9r+X3tgV/ogfMQyxWp/TOpc/XsWRZu0yIOqwfdU1Oe3X2Ami/C4+TAOU8sHkJJVNKBYwAw==" /></div>
      <button
        class="minibutton with-count js-toggler-target star-button"
        aria-label="Star this repository" title="Star long0612/ECE549-proj">
        <span class="octicon octicon-star"></span><span class="text">Star</span>
      </button>
        <a class="social-count js-social-count" href="/long0612/ECE549-proj/stargazers">
          0
        </a>
</form>  </div>

  </li>


        <li>
          <a href="/long0612/ECE549-proj/fork" class="minibutton with-count js-toggler-target fork-button lighter tooltipped-n" title="Fork your own copy of long0612/ECE549-proj to your account" aria-label="Fork your own copy of long0612/ECE549-proj to your account" rel="facebox nofollow">
            <span class="octicon octicon-git-branch-create"></span><span class="text">Fork</span>
          </a>
          <a href="/long0612/ECE549-proj/network" class="social-count">0</a>
        </li>


</ul>

        <h1 itemscope itemtype="http://data-vocabulary.org/Breadcrumb" class="entry-title public">
          <span class="repo-label"><span>public</span></span>
          <span class="mega-octicon octicon-repo"></span>
          <span class="author"><a href="/long0612" class="url fn" itemprop="url" rel="author"><span itemprop="title">long0612</span></a></span><!--
       --><span class="path-divider">/</span><!--
       --><strong><a href="/long0612/ECE549-proj" class="js-current-repository js-repo-home-link">ECE549-proj</a></strong>

          <span class="page-context-loader">
            <img alt="Octocat-spinner-32" height="16" src="https://github.global.ssl.fastly.net/images/spinners/octocat-spinner-32.gif" width="16" />
          </span>

        </h1>
      </div><!-- /.container -->
    </div><!-- /.repohead -->

    <div class="container">
      <div class="repository-with-sidebar repo-container new-discussion-timeline js-new-discussion-timeline  ">
        <div class="repository-sidebar clearfix">
            

<div class="sunken-menu vertical-right repo-nav js-repo-nav js-repository-container-pjax js-octicon-loaders">
  <div class="sunken-menu-contents">
    <ul class="sunken-menu-group">
      <li class="tooltipped tooltipped-w" aria-label="Code">
        <a href="/long0612/ECE549-proj" aria-label="Code" class="selected js-selected-navigation-item sunken-menu-item" data-hotkey="g c" data-pjax="true" data-selected-links="repo_source repo_downloads repo_commits repo_releases repo_tags repo_branches /long0612/ECE549-proj">
          <span class="octicon octicon-code"></span> <span class="full-word">Code</span>
          <img alt="Octocat-spinner-32" class="mini-loader" height="16" src="https://github.global.ssl.fastly.net/images/spinners/octocat-spinner-32.gif" width="16" />
</a>      </li>

        <li class="tooltipped tooltipped-w" aria-label="Issues">
          <a href="/long0612/ECE549-proj/issues" aria-label="Issues" class="js-selected-navigation-item sunken-menu-item js-disable-pjax" data-hotkey="g i" data-selected-links="repo_issues /long0612/ECE549-proj/issues">
            <span class="octicon octicon-issue-opened"></span> <span class="full-word">Issues</span>
            <span class='counter'>0</span>
            <img alt="Octocat-spinner-32" class="mini-loader" height="16" src="https://github.global.ssl.fastly.net/images/spinners/octocat-spinner-32.gif" width="16" />
</a>        </li>

      <li class="tooltipped tooltipped-w" aria-label="Pull Requests">
        <a href="/long0612/ECE549-proj/pulls" aria-label="Pull Requests" class="js-selected-navigation-item sunken-menu-item js-disable-pjax" data-hotkey="g p" data-selected-links="repo_pulls /long0612/ECE549-proj/pulls">
            <span class="octicon octicon-git-pull-request"></span> <span class="full-word">Pull Requests</span>
            <span class='counter'>0</span>
            <img alt="Octocat-spinner-32" class="mini-loader" height="16" src="https://github.global.ssl.fastly.net/images/spinners/octocat-spinner-32.gif" width="16" />
</a>      </li>


        <li class="tooltipped tooltipped-w" aria-label="Wiki">
          <a href="/long0612/ECE549-proj/wiki" aria-label="Wiki" class="js-selected-navigation-item sunken-menu-item js-disable-pjax" data-hotkey="g w" data-selected-links="repo_wiki /long0612/ECE549-proj/wiki">
            <span class="octicon octicon-book"></span> <span class="full-word">Wiki</span>
            <img alt="Octocat-spinner-32" class="mini-loader" height="16" src="https://github.global.ssl.fastly.net/images/spinners/octocat-spinner-32.gif" width="16" />
</a>        </li>
    </ul>
    <div class="sunken-menu-separator"></div>
    <ul class="sunken-menu-group">

      <li class="tooltipped tooltipped-w" aria-label="Pulse">
        <a href="/long0612/ECE549-proj/pulse" aria-label="Pulse" class="js-selected-navigation-item sunken-menu-item" data-pjax="true" data-selected-links="pulse /long0612/ECE549-proj/pulse">
          <span class="octicon octicon-pulse"></span> <span class="full-word">Pulse</span>
          <img alt="Octocat-spinner-32" class="mini-loader" height="16" src="https://github.global.ssl.fastly.net/images/spinners/octocat-spinner-32.gif" width="16" />
</a>      </li>

      <li class="tooltipped tooltipped-w" aria-label="Graphs">
        <a href="/long0612/ECE549-proj/graphs" aria-label="Graphs" class="js-selected-navigation-item sunken-menu-item" data-pjax="true" data-selected-links="repo_graphs repo_contributors /long0612/ECE549-proj/graphs">
          <span class="octicon octicon-graph"></span> <span class="full-word">Graphs</span>
          <img alt="Octocat-spinner-32" class="mini-loader" height="16" src="https://github.global.ssl.fastly.net/images/spinners/octocat-spinner-32.gif" width="16" />
</a>      </li>

      <li class="tooltipped tooltipped-w" aria-label="Network">
        <a href="/long0612/ECE549-proj/network" aria-label="Network" class="js-selected-navigation-item sunken-menu-item js-disable-pjax" data-selected-links="repo_network /long0612/ECE549-proj/network">
          <span class="octicon octicon-git-branch"></span> <span class="full-word">Network</span>
          <img alt="Octocat-spinner-32" class="mini-loader" height="16" src="https://github.global.ssl.fastly.net/images/spinners/octocat-spinner-32.gif" width="16" />
</a>      </li>
    </ul>


      <div class="sunken-menu-separator"></div>
      <ul class="sunken-menu-group">
        <li class="tooltipped tooltipped-w" aria-label="Settings">
          <a href="/long0612/ECE549-proj/settings" aria-label="Settings" class="js-selected-navigation-item sunken-menu-item" data-pjax="true" data-selected-links="repo_settings /long0612/ECE549-proj/settings">
            <span class="octicon octicon-tools"></span> <span class="full-word">Settings</span>
            <img alt="Octocat-spinner-32" class="mini-loader" height="16" src="https://github.global.ssl.fastly.net/images/spinners/octocat-spinner-32.gif" width="16" />
</a>        </li>
      </ul>
  </div>
</div>

              <div class="only-with-full-nav">
                

  

<div class="clone-url open"
  data-protocol-type="http"
  data-url="/users/set_protocol?protocol_selector=http&amp;protocol_type=push">
  <h3><strong>HTTPS</strong> clone URL</h3>
  <div class="clone-url-box">
    <input type="text" class="clone js-url-field"
           value="https://github.com/long0612/ECE549-proj.git" readonly="readonly">
    <span class="url-box-clippy">
    <button aria-label="copy to clipboard" class="js-zeroclipboard minibutton zeroclipboard-button" data-clipboard-text="https://github.com/long0612/ECE549-proj.git" data-copied-hint="copied!" type="button"><span class="octicon octicon-clippy"></span></button>
    </span>
  </div>
</div>

  

<div class="clone-url "
  data-protocol-type="ssh"
  data-url="/users/set_protocol?protocol_selector=ssh&amp;protocol_type=push">
  <h3><strong>SSH</strong> clone URL</h3>
  <div class="clone-url-box">
    <input type="text" class="clone js-url-field"
           value="git@github.com:long0612/ECE549-proj.git" readonly="readonly">
    <span class="url-box-clippy">
    <button aria-label="copy to clipboard" class="js-zeroclipboard minibutton zeroclipboard-button" data-clipboard-text="git@github.com:long0612/ECE549-proj.git" data-copied-hint="copied!" type="button"><span class="octicon octicon-clippy"></span></button>
    </span>
  </div>
</div>

  

<div class="clone-url "
  data-protocol-type="subversion"
  data-url="/users/set_protocol?protocol_selector=subversion&amp;protocol_type=push">
  <h3><strong>Subversion</strong> checkout URL</h3>
  <div class="clone-url-box">
    <input type="text" class="clone js-url-field"
           value="https://github.com/long0612/ECE549-proj" readonly="readonly">
    <span class="url-box-clippy">
    <button aria-label="copy to clipboard" class="js-zeroclipboard minibutton zeroclipboard-button" data-clipboard-text="https://github.com/long0612/ECE549-proj" data-copied-hint="copied!" type="button"><span class="octicon octicon-clippy"></span></button>
    </span>
  </div>
</div>


<p class="clone-options">You can clone with
      <a href="#" class="js-clone-selector" data-protocol="http">HTTPS</a>,
      <a href="#" class="js-clone-selector" data-protocol="ssh">SSH</a>,
      or <a href="#" class="js-clone-selector" data-protocol="subversion">Subversion</a>.
  <span class="help tooltipped tooltipped-n" aria-label="Get help on which URL is right for you.">
    <a href="https://help.github.com/articles/which-remote-url-should-i-use">
    <span class="octicon octicon-question"></span>
    </a>
  </span>
</p>


  <a href="github-windows://openRepo/https://github.com/long0612/ECE549-proj" class="minibutton sidebar-button" title="Save long0612/ECE549-proj to your computer and use it in GitHub Desktop." aria-label="Save long0612/ECE549-proj to your computer and use it in GitHub Desktop.">
    <span class="octicon octicon-device-desktop"></span>
    Clone in Desktop
  </a>

                <a href="/long0612/ECE549-proj/archive/master.zip"
                   class="minibutton sidebar-button"
                   aria-label="Download long0612/ECE549-proj as a zip file"
                   title="Download long0612/ECE549-proj as a zip file"
                   rel="nofollow">
                  <span class="octicon octicon-cloud-download"></span>
                  Download ZIP
                </a>
              </div>
        </div><!-- /.repository-sidebar -->

        <div id="js-repo-pjax-container" class="repository-content context-loader-container" data-pjax-container>
          

<a href="/long0612/ECE549-proj/blob/5189547ebc4893fe756eb5d934986d43b826ad6f/clustering.m" class="hidden js-permalink-shortcut" data-hotkey="y">Permalink</a>

<!-- blob contrib key: blob_contributors:v21:64a7f52e74138bca0b839718a89ade9b -->

<p title="This is a placeholder element" class="js-history-link-replace hidden"></p>

<a href="/long0612/ECE549-proj/find/master" data-pjax data-hotkey="t" class="js-show-file-finder" style="display:none">Show File Finder</a>

<div class="file-navigation">
  

<div class="select-menu js-menu-container js-select-menu" >
  <span class="minibutton select-menu-button js-menu-target" data-hotkey="w"
    data-master-branch="master"
    data-ref="master"
    role="button" aria-label="Switch branches or tags" tabindex="0" aria-haspopup="true">
    <span class="octicon octicon-git-branch"></span>
    <i>branch:</i>
    <span class="js-select-button">master</span>
  </span>

  <div class="select-menu-modal-holder js-menu-content js-navigation-container" data-pjax aria-hidden="true">

    <div class="select-menu-modal">
      <div class="select-menu-header">
        <span class="select-menu-title">Switch branches/tags</span>
        <span class="octicon octicon-remove-close js-menu-close"></span>
      </div> <!-- /.select-menu-header -->

      <div class="select-menu-filters">
        <div class="select-menu-text-filter">
          <input type="text" aria-label="Find or create a branch…" id="context-commitish-filter-field" class="js-filterable-field js-navigation-enable" placeholder="Find or create a branch…">
        </div>
        <div class="select-menu-tabs">
          <ul>
            <li class="select-menu-tab">
              <a href="#" data-tab-filter="branches" class="js-select-menu-tab">Branches</a>
            </li>
            <li class="select-menu-tab">
              <a href="#" data-tab-filter="tags" class="js-select-menu-tab">Tags</a>
            </li>
          </ul>
        </div><!-- /.select-menu-tabs -->
      </div><!-- /.select-menu-filters -->

      <div class="select-menu-list select-menu-tab-bucket js-select-menu-tab-bucket" data-tab-filter="branches">

        <div data-filterable-for="context-commitish-filter-field" data-filterable-type="substring">


            <div class="select-menu-item js-navigation-item selected">
              <span class="select-menu-item-icon octicon octicon-check"></span>
              <a href="/long0612/ECE549-proj/blob/master/clustering.m"
                 data-name="master"
                 data-skip-pjax="true"
                 rel="nofollow"
                 class="js-navigation-open select-menu-item-text js-select-button-text css-truncate-target"
                 title="master">master</a>
            </div> <!-- /.select-menu-item -->
        </div>

          <form accept-charset="UTF-8" action="/long0612/ECE549-proj/branches" class="js-create-branch select-menu-item select-menu-new-item-form js-navigation-item js-new-item-form" method="post"><div style="margin:0;padding:0;display:inline"><input name="authenticity_token" type="hidden" value="i/bjcwUjGCCeJMboHiDXu5fGnBDB3wJk50K5r3Y1Unx3/rCzKerMeR88qKBlYp2/FfGSK7jLVCiZl9Ma//IciA==" /></div>
            <span class="octicon octicon-git-branch-create select-menu-item-icon"></span>
            <div class="select-menu-item-text">
              <h4>Create branch: <span class="js-new-item-name"></span></h4>
              <span class="description">from ‘master’</span>
            </div>
            <input type="hidden" name="name" id="name" class="js-new-item-value">
            <input type="hidden" name="branch" id="branch" value="master" />
            <input type="hidden" name="path" id="path" value="clustering.m" />
          </form> <!-- /.select-menu-item -->

      </div> <!-- /.select-menu-list -->

      <div class="select-menu-list select-menu-tab-bucket js-select-menu-tab-bucket" data-tab-filter="tags">
        <div data-filterable-for="context-commitish-filter-field" data-filterable-type="substring">


        </div>

        <div class="select-menu-no-results">Nothing to show</div>
      </div> <!-- /.select-menu-list -->

    </div> <!-- /.select-menu-modal -->
  </div> <!-- /.select-menu-modal-holder -->
</div> <!-- /.select-menu -->

  <div class="breadcrumb">
    <span class='repo-root js-repo-root'><span itemscope="" itemtype="http://data-vocabulary.org/Breadcrumb"><a href="/long0612/ECE549-proj" data-branch="master" data-direction="back" data-pjax="true" itemscope="url"><span itemprop="title">ECE549-proj</span></a></span></span><span class="separator"> / </span><strong class="final-path">clustering.m</strong> <button aria-label="copy to clipboard" class="js-zeroclipboard minibutton zeroclipboard-button" data-clipboard-text="clustering.m" data-copied-hint="copied!" type="button"><span class="octicon octicon-clippy"></span></button>
  </div>
</div>


  <div class="commit commit-loader file-history-tease js-deferred-content" data-url="/long0612/ECE549-proj/contributors/master/clustering.m">
    Fetching contributors…

    <div class="participation">
      <p class="loader-loading"><img alt="Octocat-spinner-32-eaf2f5" height="16" src="https://github.global.ssl.fastly.net/images/spinners/octocat-spinner-32-EAF2F5.gif" width="16" /></p>
      <p class="loader-error">Cannot retrieve contributors at this time</p>
    </div>
  </div>

<div class="file-box">
  <div class="file">
    <div class="meta clearfix">
      <div class="info file-name">
        <span class="icon"><b class="octicon octicon-file-text"></b></span>
        <span class="mode" title="File Mode">file</span>
        <span class="meta-divider"></span>
          <span>57 lines (42 sloc)</span>
          <span class="meta-divider"></span>
        <span>1.269 kb</span>
      </div>
      <div class="actions">
        <div class="button-group">
            <a class="minibutton tooltipped tooltipped-w"
               href="github-windows://openRepo/https://github.com/long0612/ECE549-proj?branch=master&amp;filepath=clustering.m" aria-label="Open this file in GitHub for Windows">
                <span class="octicon octicon-device-desktop"></span> Open
            </a>
                <a class="minibutton js-update-url-with-hash"
                   href="/long0612/ECE549-proj/edit/master/clustering.m"
                   data-method="post" rel="nofollow" data-hotkey="e">Edit</a>
          <a href="/long0612/ECE549-proj/raw/master/clustering.m" class="button minibutton " id="raw-url">Raw</a>
            <a href="/long0612/ECE549-proj/blame/master/clustering.m" class="button minibutton js-update-url-with-hash">Blame</a>
          <a href="/long0612/ECE549-proj/commits/master/clustering.m" class="button minibutton " rel="nofollow">History</a>
        </div><!-- /.button-group -->

            <a class="minibutton danger empty-icon"
               href="/long0612/ECE549-proj/delete/master/clustering.m"
               data-method="post" data-test-id="delete-blob-file" rel="nofollow">

          Delete
        </a>
      </div><!-- /.actions -->
    </div>
        <div class="blob-wrapper data type-matlab js-blob-data">
        <table class="file-code file-diff tab-size-8">
          <tr class="file-code-line">
            <td class="blob-line-nums">
              <span id="L1" rel="#L1">1</span>
<span id="L2" rel="#L2">2</span>
<span id="L3" rel="#L3">3</span>
<span id="L4" rel="#L4">4</span>
<span id="L5" rel="#L5">5</span>
<span id="L6" rel="#L6">6</span>
<span id="L7" rel="#L7">7</span>
<span id="L8" rel="#L8">8</span>
<span id="L9" rel="#L9">9</span>
<span id="L10" rel="#L10">10</span>
<span id="L11" rel="#L11">11</span>
<span id="L12" rel="#L12">12</span>
<span id="L13" rel="#L13">13</span>
<span id="L14" rel="#L14">14</span>
<span id="L15" rel="#L15">15</span>
<span id="L16" rel="#L16">16</span>
<span id="L17" rel="#L17">17</span>
<span id="L18" rel="#L18">18</span>
<span id="L19" rel="#L19">19</span>
<span id="L20" rel="#L20">20</span>
<span id="L21" rel="#L21">21</span>
<span id="L22" rel="#L22">22</span>
<span id="L23" rel="#L23">23</span>
<span id="L24" rel="#L24">24</span>
<span id="L25" rel="#L25">25</span>
<span id="L26" rel="#L26">26</span>
<span id="L27" rel="#L27">27</span>
<span id="L28" rel="#L28">28</span>
<span id="L29" rel="#L29">29</span>
<span id="L30" rel="#L30">30</span>
<span id="L31" rel="#L31">31</span>
<span id="L32" rel="#L32">32</span>
<span id="L33" rel="#L33">33</span>
<span id="L34" rel="#L34">34</span>
<span id="L35" rel="#L35">35</span>
<span id="L36" rel="#L36">36</span>
<span id="L37" rel="#L37">37</span>
<span id="L38" rel="#L38">38</span>
<span id="L39" rel="#L39">39</span>
<span id="L40" rel="#L40">40</span>
<span id="L41" rel="#L41">41</span>
<span id="L42" rel="#L42">42</span>
<span id="L43" rel="#L43">43</span>
<span id="L44" rel="#L44">44</span>
<span id="L45" rel="#L45">45</span>
<span id="L46" rel="#L46">46</span>
<span id="L47" rel="#L47">47</span>
<span id="L48" rel="#L48">48</span>
<span id="L49" rel="#L49">49</span>
<span id="L50" rel="#L50">50</span>
<span id="L51" rel="#L51">51</span>
<span id="L52" rel="#L52">52</span>
<span id="L53" rel="#L53">53</span>
<span id="L54" rel="#L54">54</span>
<span id="L55" rel="#L55">55</span>
<span id="L56" rel="#L56">56</span>

            </td>
            <td class="blob-line-code"><div class="code-body highlight"><pre><div class='line' id='LC1'><span class="c">%% Blob detection stuff</span></div><div class='line' id='LC2'><br/></div><div class='line' id='LC3'><span class="n">sigma</span> <span class="p">=</span> <span class="mi">2</span><span class="p">;</span></div><div class='line' id='LC4'><br/></div><div class='line' id='LC5'><span class="n">thresList</span> <span class="p">=</span> <span class="p">[</span><span class="mf">0.03</span> <span class="mf">0.009</span> <span class="mf">0.009</span> <span class="mf">0.003</span> <span class="mf">0.008</span> <span class="mf">0.003</span> <span class="mf">0.005</span> <span class="mf">0.008</span><span class="p">];</span></div><div class='line' id='LC6'><span class="n">alphaList</span> <span class="p">=</span> <span class="p">[</span><span class="mf">0.18</span> <span class="mf">0.06</span> <span class="mf">0.1</span> <span class="mf">0.06</span> <span class="mf">0.09</span> <span class="mf">0.2</span> <span class="mf">0.18</span> <span class="mf">0.1</span><span class="p">];</span></div><div class='line' id='LC7'><br/></div><div class='line' id='LC8'><span class="n">alpha</span> <span class="p">=</span> <span class="n">alphaList</span><span class="p">(</span><span class="mi">4</span><span class="p">);</span></div><div class='line' id='LC9'><span class="n">thres</span> <span class="p">=</span> <span class="n">thresList</span><span class="p">(</span><span class="mi">4</span><span class="p">);</span></div><div class='line' id='LC10'><span class="n">scale</span> <span class="p">=</span> <span class="p">[</span><span class="mi">1</span> <span class="mi">2</span> <span class="mi">3</span> <span class="mi">4</span> <span class="mi">5</span> <span class="mi">10</span><span class="p">]</span><span class="o">/</span><span class="nb">sqrt</span><span class="p">(</span><span class="mi">2</span><span class="p">);</span></div><div class='line' id='LC11'><br/></div><div class='line' id='LC12'><span class="n">vid</span> <span class="p">=</span> <span class="n">VideoReader</span><span class="p">(</span><span class="s">&#39;charade.mp4&#39;</span><span class="p">);</span></div><div class='line' id='LC13'><span class="n">imgs</span> <span class="p">=</span> <span class="n">cell</span><span class="p">(</span><span class="mi">1</span><span class="p">);</span></div><div class='line' id='LC14'><span class="n">samples</span> <span class="p">=</span> <span class="mi">1</span><span class="p">;</span></div><div class='line' id='LC15'><br/></div><div class='line' id='LC16'><span class="c">%% Very specific scene for this video.</span></div><div class='line' id='LC17'><br/></div><div class='line' id='LC18'><span class="k">for</span> <span class="n">sample</span> <span class="p">=</span> <span class="mi">1300</span><span class="p">:</span> <span class="mi">30</span><span class="p">:</span><span class="mi">2710</span></div><div class='line' id='LC19'>&nbsp;&nbsp;&nbsp;&nbsp;<span class="n">imgs</span><span class="p">{</span><span class="n">samples</span><span class="p">}</span> <span class="p">=</span> <span class="n">read</span><span class="p">(</span><span class="n">vid</span><span class="p">,</span> <span class="n">sample</span><span class="p">);</span></div><div class='line' id='LC20'>&nbsp;&nbsp;&nbsp;&nbsp;<span class="n">samples</span> <span class="p">=</span> <span class="n">samples</span> <span class="o">+</span> <span class="mi">1</span><span class="p">;</span></div><div class='line' id='LC21'><span class="k">end</span></div><div class='line' id='LC22'><br/></div><div class='line' id='LC23'><span class="c">%% Run blob detection on each of the 48 frame</span></div><div class='line' id='LC24'><br/></div><div class='line' id='LC25'><span class="n">Cx</span> <span class="p">=</span> <span class="n">cell</span><span class="p">(</span><span class="mi">1</span><span class="p">);</span></div><div class='line' id='LC26'><span class="n">Cy</span> <span class="p">=</span> <span class="n">cell</span><span class="p">(</span><span class="mi">1</span><span class="p">);</span></div><div class='line' id='LC27'><span class="n">Cr</span> <span class="p">=</span> <span class="n">cell</span><span class="p">(</span><span class="mi">1</span><span class="p">);</span></div><div class='line' id='LC28'><span class="n">sifts</span> <span class="p">=</span> <span class="n">cell</span><span class="p">(</span><span class="mi">1</span><span class="p">);</span></div><div class='line' id='LC29'><br/></div><div class='line' id='LC30'><span class="n">descriptors</span> <span class="p">=</span> <span class="p">[];</span></div><div class='line' id='LC31'><br/></div><div class='line' id='LC32'><span class="k">for</span> <span class="nb">i</span> <span class="p">=</span> <span class="mi">1</span><span class="p">:</span><span class="n">samples</span><span class="o">-</span><span class="mi">1</span></div><div class='line' id='LC33'>&nbsp;&nbsp;&nbsp;&nbsp;<span class="n">im</span> <span class="p">=</span> <span class="n">rgb2gray</span><span class="p">(</span><span class="n">im2double</span><span class="p">(</span><span class="n">imgs</span><span class="p">{</span><span class="nb">i</span><span class="p">}));</span></div><div class='line' id='LC34'>&nbsp;&nbsp;&nbsp;&nbsp;<span class="p">[</span><span class="n">Cx</span><span class="p">{</span><span class="nb">i</span><span class="p">},</span> <span class="n">Cy</span><span class="p">{</span><span class="nb">i</span><span class="p">},</span> <span class="n">Cr</span><span class="p">{</span><span class="nb">i</span><span class="p">},</span> <span class="n">a</span><span class="p">,</span> <span class="n">b</span><span class="p">,</span> <span class="n">phi</span><span class="p">]</span> <span class="p">=</span> <span class="n">blobDetectAffine</span><span class="p">(</span><span class="n">im</span><span class="p">,</span> <span class="n">sigma</span><span class="p">,</span> <span class="n">scale</span><span class="p">,</span> <span class="n">thres</span><span class="p">,</span> <span class="n">alpha</span><span class="p">);</span></div><div class='line' id='LC35'>&nbsp;&nbsp;&nbsp;&nbsp;<span class="n">sifts</span><span class="p">{</span><span class="nb">i</span><span class="p">}</span> <span class="p">=</span> <span class="n">find_sift</span><span class="p">(</span><span class="n">im</span><span class="p">,</span> <span class="p">[</span><span class="n">Cx</span><span class="p">{</span><span class="nb">i</span><span class="p">},</span> <span class="n">Cy</span><span class="p">{</span><span class="nb">i</span><span class="p">},</span> <span class="n">Cr</span><span class="p">{</span><span class="nb">i</span><span class="p">}],</span> <span class="mf">1.5</span><span class="p">);</span></div><div class='line' id='LC36'>&nbsp;&nbsp;&nbsp;&nbsp;<span class="n">descriptors</span> <span class="p">=</span> <span class="p">[</span><span class="n">descriptors</span><span class="p">;</span> <span class="n">sifts</span><span class="p">{</span><span class="nb">i</span><span class="p">}];</span></div><div class='line' id='LC37'><span class="k">end</span></div><div class='line' id='LC38'><br/></div><div class='line' id='LC39'><span class="n">words</span> <span class="p">=</span> <span class="n">cell</span><span class="p">(</span><span class="mi">1</span><span class="p">);</span></div><div class='line' id='LC40'><br/></div><div class='line' id='LC41'><span class="c">%% Work needs to be done to calculate the optimum k.</span></div><div class='line' id='LC42'><span class="c">%% I was thinking about varying it from 4 through 8 or so then</span></div><div class='line' id='LC43'><span class="c">%% use Elbow method to get the best K. For now k = 5.</span></div><div class='line' id='LC44'><br/></div><div class='line' id='LC45'><span class="n">k</span> <span class="p">=</span> <span class="mi">5</span><span class="p">;</span></div><div class='line' id='LC46'><span class="c">%% This uses Euclidean distance.</span></div><div class='line' id='LC47'><span class="n">id2clusters</span> <span class="p">=</span> <span class="n">kmeans</span><span class="p">(</span><span class="n">descriptors</span><span class="p">,</span> <span class="n">k</span><span class="p">);</span></div><div class='line' id='LC48'><br/></div><div class='line' id='LC49'><span class="k">for</span> <span class="nb">i</span> <span class="p">=</span> <span class="mi">1</span><span class="p">:</span><span class="n">samples</span><span class="o">-</span><span class="mi">1</span></div><div class='line' id='LC50'>&nbsp;&nbsp;&nbsp;&nbsp;<span class="k">if</span> <span class="nb">i</span> <span class="o">==</span> <span class="mi">1</span></div><div class='line' id='LC51'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span class="n">c</span> <span class="p">=</span> <span class="n">id2clusters</span><span class="p">(</span><span class="nb">find</span><span class="p">(</span><span class="n">Cx</span><span class="p">{</span><span class="nb">i</span><span class="p">}));</span></div><div class='line' id='LC52'>&nbsp;&nbsp;&nbsp;&nbsp;<span class="k">else</span></div><div class='line' id='LC53'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span class="n">c</span> <span class="p">=</span> <span class="n">id2clusters</span><span class="p">(</span><span class="nb">find</span><span class="p">(</span><span class="n">Cx</span><span class="p">{</span><span class="nb">i</span><span class="p">})</span><span class="o">+</span><span class="nb">size</span><span class="p">(</span><span class="n">Cx</span><span class="p">{</span><span class="nb">i</span><span class="o">-</span><span class="mi">1</span><span class="p">},</span><span class="mi">1</span><span class="p">));</span></div><div class='line' id='LC54'>&nbsp;&nbsp;&nbsp;&nbsp;<span class="k">end</span></div><div class='line' id='LC55'>&nbsp;&nbsp;&nbsp;&nbsp;<span class="n">words</span><span class="p">{</span><span class="nb">i</span><span class="p">}</span> <span class="p">=</span> <span class="n">histc</span><span class="p">(</span><span class="n">c</span><span class="o">&#39;</span><span class="p">,</span> <span class="nb">linspace</span><span class="p">(</span><span class="mi">1</span><span class="p">,</span><span class="n">k</span><span class="p">,</span><span class="n">k</span><span class="p">));</span></div><div class='line' id='LC56'><span class="k">end</span></div></pre></div></td>
          </tr>
        </table>
  </div>

  </div>
</div>

<a href="#jump-to-line" rel="facebox[.linejump]" data-hotkey="l" class="js-jump-to-line" style="display:none">Jump to Line</a>
<div id="jump-to-line" style="display:none">
  <form accept-charset="UTF-8" class="js-jump-to-line-form">
    <input class="linejump-input js-jump-to-line-field" type="text" placeholder="Jump to line&hellip;" autofocus>
    <button type="submit" class="button">Go</button>
  </form>
</div>

        </div>

      </div><!-- /.repo-container -->
      <div class="modal-backdrop"></div>
    </div><!-- /.container -->
  </div><!-- /.site -->


    </div><!-- /.wrapper -->

      <div class="container">
  <div class="site-footer">
    <ul class="site-footer-links right">
      <li><a href="https://status.github.com/">Status</a></li>
      <li><a href="http://developer.github.com">API</a></li>
      <li><a href="http://training.github.com">Training</a></li>
      <li><a href="http://shop.github.com">Shop</a></li>
      <li><a href="/blog">Blog</a></li>
      <li><a href="/about">About</a></li>

    </ul>

    <a href="/">
      <span class="mega-octicon octicon-mark-github" title="GitHub"></span>
    </a>

    <ul class="site-footer-links">
      <li>&copy; 2014 <span title="0.07533s from github-fe120-cp1-prd.iad.github.net">GitHub</span>, Inc.</li>
        <li><a href="/site/terms">Terms</a></li>
        <li><a href="/site/privacy">Privacy</a></li>
        <li><a href="/security">Security</a></li>
        <li><a href="/contact">Contact</a></li>
    </ul>
  </div><!-- /.site-footer -->
</div><!-- /.container -->


    <div class="fullscreen-overlay js-fullscreen-overlay" id="fullscreen_overlay">
  <div class="fullscreen-container js-fullscreen-container">
    <div class="textarea-wrap">
      <textarea name="fullscreen-contents" id="fullscreen-contents" class="fullscreen-contents js-fullscreen-contents" placeholder="" data-suggester="fullscreen_suggester"></textarea>
    </div>
  </div>
  <div class="fullscreen-sidebar">
    <a href="#" class="exit-fullscreen js-exit-fullscreen tooltipped tooltipped-w" aria-label="Exit Zen Mode">
      <span class="mega-octicon octicon-screen-normal"></span>
    </a>
    <a href="#" class="theme-switcher js-theme-switcher tooltipped tooltipped-w"
      aria-label="Switch themes">
      <span class="octicon octicon-color-mode"></span>
    </a>
  </div>
</div>



    <div id="ajax-error-message" class="flash flash-error">
      <span class="octicon octicon-alert"></span>
      <a href="#" class="octicon octicon-remove-close close js-ajax-error-dismiss"></a>
      Something went wrong with that request. Please try again.
    </div>


      <script crossorigin="anonymous" src="https://github.global.ssl.fastly.net/assets/frameworks-0761ba432b838d086e553e65a1be483eca0cba97.js" type="text/javascript"></script>
      <script async="async" crossorigin="anonymous" src="https://github.global.ssl.fastly.net/assets/github-320bd0f5b22fb60db7de2f691e6a8956971f5da2.js" type="text/javascript"></script>
      
      
  </body>
</html>

