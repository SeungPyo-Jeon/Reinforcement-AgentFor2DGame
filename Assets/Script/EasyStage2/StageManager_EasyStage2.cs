using System;
using System.Collections.Generic;
using System.Text.RegularExpressions;
using UnityEngine;
using UnityEngine.SceneManagement;

public class StageManager_EasyStage2 : MonoBehaviour
{
    [SerializeField]
    public GameObject CoinPrefab;
    [SerializeField]
    public List<Vector2> CoinPosList;
    public int CoinCount;
    protected bool m_IsSafe;
    public GameObject Player;
    public GameObject SafetyZone;
    public string resetMode = "";
    private float distance;

    // Start is called once before the first execution of Update after the MonoBehaviour is created
    void Start()
    {
        CoinCount = 0;
        for (int i = 0; i < CoinPosList.Count; i++)
        {
            GameObject coin = Instantiate(CoinPrefab, CoinPosList[i], Quaternion.identity);
            coin.transform.SetParent(transform);
        }
        Player = GameObject.Find("Player");
        SafetyZone = GameObject.Find("save_tile");
        distance = Vector2.Distance(Player.transform.position, SafetyZone.transform.position);
    }

    // Update is called once per frame
    void FixedUpdate()
    {

    }
    public void EnterSafetyZone()
    {
        if (CoinCount == CoinPosList.Count)
        {
            SceneManager.LoadScene("SampleScene");
        }
    }
    public void EnterCoin(Collider2D collider)
    {
        CoinCount++;
        collider.gameObject.SetActive(false);
        Destroy(collider.gameObject);
    }

    public void Death()
    {
        SceneManager.LoadScene("SampleScene");
    }
    //Player Agnet
    public void Reset()
    {
        GameObject[] gameObjects;
        gameObjects = GameObject.FindGameObjectsWithTag("Coin");
        for (int i = 0; i < gameObjects.Length; i++)
        {
            Destroy(gameObjects[i]);
        }

        CoinCount = 0;
        for (int i = 0; i < CoinPosList.Count; i++)
        {
            GameObject coin = Instantiate(CoinPrefab, CoinPosList[i], Quaternion.identity);
            coin.transform.SetParent(transform);
        }
        if (resetMode == "EaseStage2")
        {
            Player.transform.position = new Vector3(UnityEngine.Random.Range(1, 20), UnityEngine.Random.Range(1, 12), 0);
            do
            {
                SafetyZone.transform.position = new Vector3(UnityEngine.Random.Range(1, 12), UnityEngine.Random.Range(1, 12), 0);
            } while (Vector2.Distance(Player.transform.position, SafetyZone.transform.position) < 3);

        }
        else
        {
            Player.transform.position = Player.gameObject.GetComponent<PlayerAgent_EasyStage2>().initPos;
        }
        distance = Vector2.Distance(Player.transform.position, SafetyZone.transform.position);
    }

    public float EnterSafetyZone_Agent()
    {
        if (CoinCount == CoinPosList.Count)
        {
            return 1.0f;
        }
        return -1.0f;
    }
    public int getCoinCount()
    {
        return CoinCount;
    }
    public float getDistance(float positive_scale, float negative_scale)
    {
        float temp = distance;
        distance = Vector2.Distance(Player.transform.position, SafetyZone.transform.position);
        if (temp > distance)
        {
            Debug.Log("Distance: " + (temp - distance));
            return (temp - distance) * positive_scale;
        }
        else if (temp < distance)
        {
            Debug.Log("Distance: " + (temp - distance));
            return (temp - distance) * negative_scale;
        }
        else if (temp == distance)
        {
            return -(float)Math.Pow(1 - (1 / Vector2.Distance(Player.transform.position, SafetyZone.transform.position)), 2);
        }
        return 0;
    }
}